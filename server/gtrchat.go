package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/tokenizer"
	typemodel "github.com/ollama/ollama/types/model"
)

func (s *Server) GTRChatHandler(c *gin.Context) {
	var req api.GTRChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		if err == io.EOF {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		} else {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		}
		return
	}

	caps := []typemodel.Capability{typemodel.CapabilityCompletion}
	r, m, opts, err := s.scheduleRunner(c.Request.Context(), req.Model, caps, req.Options, req.KeepAlive)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	if m.Config.Renderer != "gemma4" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "GTRChat is currently only supported for Gemma 4 models"})
		return
	}

	builder, err := s.newGTRPromptBuilder(m.ModelPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to initialize GTR prompt builder: %v", err)})
		return
	}

	promptTokens, images, err := builder.Build(c.Request.Context(), req.Turns)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to build GTR prompt: %v", err)})
		return
	}

	prompt, err := builder.SafeDetokenize(c.Request.Context(), r, promptTokens)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to safely detokenize GTR prompt: %v", err)})
		return
	}

	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	ch := make(chan any)
	go func() {
		defer close(ch)

		parser := &GTRResponseParser{
			streamMode: req.StreamMode,
			onEvent: func(event api.GTRChatResponseEvent) {
				ch <- event
			},
		}

		err := r.Completion(c.Request.Context(), llm.CompletionRequest{
			Prompt:       prompt,
			PromptTokens: promptTokens,
			Images:       images,
			Options:      opts,
		}, func(cr llm.CompletionResponse) {
			parser.Parse(cr)
		})
		if err != nil {
			ch <- gin.H{"error": err.Error()}
		}
	}()

	if stream {
		s.streamResponse(c, ch)
	} else {
		var fullResponse api.GTRChatResponse
		fullResponse.Model = req.Model
		fullResponse.CreatedAt = time.Now().UTC()
		
		for e := range ch {
			if ev, ok := e.(api.GTRChatResponseEvent); ok {
				if ev.Type == "done" {
					fullResponse.Done = true
				}
				// Optionally aggregate content into turns here if needed
			} else if h, ok := e.(gin.H); ok {
				c.JSON(http.StatusInternalServerError, h)
				return
			}
		}
		c.JSON(http.StatusOK, fullResponse)
	}
}

func (s *Server) streamResponse(c *gin.Context, ch <-chan any) {
	c.Header("Content-Type", "application/json")
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		bts, err := json.Marshal(val)
		if err != nil {
			slog.Error("failed to marshal GTR event", "error", err)
			return true
		}

		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			slog.Error("failed to write GTR event to client", "error", err)
			return false
		}

		return true
	})
}

type GTRResponseParser struct {
	streamMode string
	onEvent    func(api.GTRChatResponseEvent)
	
	state      string // "", "thinking", "tool_call"
	pending    string // buffer for partial markers
}

func (p *GTRResponseParser) Parse(cr llm.CompletionResponse) {
	if p.streamMode == "raw" {
		p.onEvent(api.GTRChatResponseEvent{Type: "text", Content: cr.Content})
		if cr.Done {
			p.onEvent(api.GTRChatResponseEvent{Type: "done", Status: "complete"})
		}
		return
	}

	// Structured mode: State machine based on strings
	// In a production environment, this would handle token-level transitions more robustly
	p.pending += cr.Content

	for {
		found := false
		
		// 1. Check for thinking start
		if idx := strings.Index(p.pending, "<|channel>thought\n"); idx != -1 {
			if idx > 0 {
				p.emit(p.pending[:idx])
			}
			p.state = "thinking"
			p.pending = p.pending[idx+len("<|channel>thought\n"):]
			found = true
		}
		
		// 2. Check for thinking end
		if idx := strings.Index(p.pending, "<channel|>"); idx != -1 {
			if idx > 0 {
				p.emit(p.pending[:idx])
			}
			p.state = ""
			p.pending = p.pending[idx+len("<channel|>"):]
			found = true
		}

		// 3. Check for tool call start
		if idx := strings.Index(p.pending, "<|tool_call>"); idx != -1 {
			if idx > 0 {
				p.emit(p.pending[:idx])
			}
			p.pending = p.pending[idx+len("<|tool_call>"):]
			p.state = "tool_call"
			found = true
		}

		// 4. Check for tool call content (call:name{)
		if p.state == "tool_call" && strings.Contains(p.pending, "call:") && strings.Contains(p.pending, "{") {
			idxCall := strings.Index(p.pending, "call:")
			idxBrace := strings.Index(p.pending, "{")
			if idxBrace > idxCall {
				name := p.pending[idxCall+len("call:") : idxBrace]
				p.onEvent(api.GTRChatResponseEvent{Type: "tool_call_start", Name: name})
				p.pending = p.pending[idxBrace+1:]
				found = true
			}
		}

		// 5. Check for tool call end
		if idx := strings.Index(p.pending, "<tool_call|>"); idx != -1 {
			if idx > 0 {
				p.emit(p.pending[:idx])
			}
			p.onEvent(api.GTRChatResponseEvent{Type: "tool_call_end"})
			p.state = ""
			p.pending = p.pending[idx+len("<tool_call|>"):]
			found = true
		}

		// 6. Check for turn end / tool response
		if strings.Contains(p.pending, "<turn|>") || strings.Contains(p.pending, "<|tool_response>") {
			// Stop processing and let 'Done' handle it
			break
		}

		if !found {
			break
		}
	}

	// Emit whatever is left if it's not a partial marker
	// We keep a small safety buffer for markers
	if len(p.pending) > 20 {
		safeLen := len(p.pending) - 20
		p.emit(p.pending[:safeLen])
		p.pending = p.pending[safeLen:]
	}

	if cr.Done {
		// Emit final pending
		if p.pending != "" {
			p.emit(p.pending)
		}
		status := "complete"
		if strings.Contains(cr.Content, "<|tool_response>") {
			status = "call_wait"
		}
		p.onEvent(api.GTRChatResponseEvent{Type: "done", Status: status})
	}
}

func (p *GTRResponseParser) emit(content string) {
	if content == "" {
		return
	}
	contentType := "text"
	switch p.state {
	case "thinking":
		contentType = "thinking"
	case "tool_call":
		contentType = "tool_call_delta"
	}
	p.onEvent(api.GTRChatResponseEvent{Type: contentType, Content: content})
}

type GTRPromptBuilder struct {
	tokenizer tokenizer.Tokenizer
	images    []llm.ImageData
}

func (s *Server) newGTRPromptBuilder(modelPath string) (*GTRPromptBuilder, error) {
	tp, err := model.NewTextProcessor(modelPath)
	if err != nil {
		return nil, err
	}
	return &GTRPromptBuilder{tokenizer: tp}, nil
}

func (b *GTRPromptBuilder) Build(ctx context.Context, turns []api.GTRChatTurn) ([]int, []llm.ImageData, error) {
	var fullTokens []int32

	for i, turn := range turns {
		// Turn Start: <|turn>role\n
		turnHeader := fmt.Sprintf("<|turn>%s\n", turn.Role)
		tokens, err := b.tokenizer.Encode(turnHeader, false)
		if err != nil {
			return nil, nil, err
		}
		fullTokens = append(fullTokens, tokens...)

		for _, comp := range turn.Components {
			compTokens, err := b.encodeComponent(comp)
			if err != nil {
				return nil, nil, err
			}
			fullTokens = append(fullTokens, compTokens...)
		}

		// Turn End: <turn|>
		// Note: The model turn usually doesn't have <turn|> if we are waiting for output,
		// but the request might provide historical turns that DO have it.
		// For the last turn, if it's a model turn, we skip <turn|> to allow continuation.
		if i < len(turns)-1 || turn.Role != "model" {
			tokens, err = b.tokenizer.Encode("<turn|>", false)
			if err != nil {
				return nil, nil, err
			}
			fullTokens = append(fullTokens, tokens...)
		}
	}

	// Ensure model turn start if the last turn was not a model turn? 
	// Or let the client provide the start of the model turn.
	// Actually, the prompt should end with the start of the model turn for the next generation.
	// If the last turn role is NOT "model", we should probably append a model turn start.
	if len(turns) > 0 && turns[len(turns)-1].Role != "model" {
		tokens, _ := b.tokenizer.Encode("<|turn>model\n", false)
		fullTokens = append(fullTokens, tokens...)
	}

	res := make([]int, len(fullTokens))
	for i, t := range fullTokens {
		res[i] = int(t)
	}
	return res, b.images, nil
}

// SafeDetokenize decodes tokens but replaces our protected image token ID (258880)
// with the [img-N] tags expected by the runner. This prevents literal text in 
// components from being misinterpreted as image markers.
func (b *GTRPromptBuilder) SafeDetokenize(ctx context.Context, r llm.LlamaServer, tokens []int) (string, error) {
	// With the new token pass-through runner, we no longer need to perform 
	// special detokenization for image placement. We return a standard 
	// detokenized string for logging and debugging purposes.
	return r.Detokenize(ctx, tokens)
}

var ALL_STRUCTURAL = []string{
	"<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
	"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
	"<|tool_response>", "<tool_response|>", "<|\"|>",
	"<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>",
	"<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>",
}

func (b *GTRPromptBuilder) encodeComponent(comp api.GTRChatComponent) ([]int32, error) {
	switch comp.CType {
	case "systemtext", "answer", "thinking":
		var data api.GTRTextData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}
		// String components are encoded with an empty allowed list, ensuring
		// that they can never produce structural control tokens.
		if comp.CType == "thinking" {
			start, _ := b.tokenizer.Encode("<|channel>thought\n", false)
			content, _ := b.tokenizer.EncodeWithAllowed(data.Text, false, []string{})
			end, _ := b.tokenizer.Encode("<channel|>", false)
			return append(append(start, content...), end...), nil
		}
		return b.tokenizer.EncodeWithAllowed(data.Text, false, []string{})

	case "toolschema":
		var data api.GTRToolSchemaData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}
		var tokens []int32
		for _, tool := range data.Tools {
			t1, _ := b.tokenizer.Encode("<|tool>", false)
			// Encode tool JSON schema or simplified format
			// The requirements say for toolschema, values are encoded with allowedTokens: []
			// Framing is raw IDs.
			// Simplified framing for example: call:name{...}
			// But for schema, it's different. Let's use a generic JSON-like framing.
			schemaStr := fmt.Sprintf("{\"name\": \"%s\", \"description\": \"%s\"}", tool.Name, tool.Description)
			t2, _ := b.tokenizer.EncodeWithAllowed(schemaStr, false, []string{}) 
			t3, _ := b.tokenizer.Encode("<tool|>", false)
			tokens = append(append(append(tokens, t1...), t2...), t3...)
		}
		return tokens, nil

	case "toolcall":
		var data api.GTRToolCallData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}
		// <|tool_call>call:name{key:<|"|>val<|"|>,...}<tool_call|>
		t1, _ := b.tokenizer.Encode("<|tool_call>", false)
		t2, _ := b.tokenizer.EncodeWithAllowed(fmt.Sprintf("call:%s{", data.Name), false, ALL_STRUCTURAL)
		tokens := append(t1, t2...)
		for i, arg := range data.Args {
			if i > 0 {
				sep, _ := b.tokenizer.EncodeWithAllowed(",", false, ALL_STRUCTURAL)
				tokens = append(tokens, sep...)
			}
			k, _ := b.tokenizer.EncodeWithAllowed(arg.Key, false, []string{})
			c, _ := b.tokenizer.EncodeWithAllowed(":", false, []string{})
			tokens = append(append(tokens, k...), c...)
			
			q1, _ := b.tokenizer.Encode("<|\"|>", false)
			v, _ := b.tokenizer.EncodeWithAllowed(arg.Val, false, []string{})
			q2, _ := b.tokenizer.Encode("<|\"|>", false)
			tokens = append(append(append(tokens, q1...), v...), q2...)
		}
		endBrace, _ := b.tokenizer.EncodeWithAllowed("}", false, ALL_STRUCTURAL)
		tEnd, _ := b.tokenizer.Encode("<tool_call|>", false)
		return append(append(tokens, endBrace...), tEnd...), nil

	case "tool_response":
		var data api.GTRToolCallData // Reuse toolcall shape for response result
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}
		t1, _ := b.tokenizer.Encode("<|tool_response>", false)
		// Assuming tool response follows a similar pattern or just raw result
		t2, _ := b.tokenizer.EncodeWithAllowed(fmt.Sprintf("{\"%s\": ", data.Name), false, []string{})
		tokens := append(t1, t2...)
		q1, _ := b.tokenizer.Encode("<|\"|>", false)
		// The result is usually in data.Args[0].Val if we follow the weather example
		val := ""
		if len(data.Args) > 0 { val = data.Args[0].Val }
		v, _ := b.tokenizer.EncodeWithAllowed(val, false, []string{})
		q2, _ := b.tokenizer.Encode("<|\"|>", false)
		t3, _ := b.tokenizer.EncodeWithAllowed("}", false, []string{})
		tEnd, _ := b.tokenizer.Encode("<tool_response|>", false)
		return append(append(append(append(append(tokens, q1...), v...), q2...), t3...), tEnd...), nil

	case "image":
		var data api.GTRMultimodalData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}
		// We use the official <|image|> structural token ID (258880) as a marker.
		// Since user text is encoded with allowedTokens=[], the tokenizer will 
		// never produce this ID from user-provided string content, making it 
		// a safe and unambiguous placeholder.
		imgData, err := base64.StdEncoding.DecodeString(data.Multimodal)
		if err != nil {
			return nil, fmt.Errorf("failed to decode image base64: %v", err)
		}
		b.images = append(b.images, llm.ImageData{Data: imgData})
		
		return []int32{258880}, nil

	case "tool":
		start, _ := b.tokenizer.Encode("<|tool>", false)
		content, _ := b.tokenizer.EncodeWithAllowed(string(comp.Data), false, []string{})
		end, _ := b.tokenizer.Encode("<tool|>", false)
		return append(append(start, content...), end...), nil

	case "call":
		start, _ := b.tokenizer.Encode("<|call>", false)
		content, _ := b.tokenizer.EncodeWithAllowed(string(comp.Data), false, []string{})
		end, _ := b.tokenizer.Encode("<call|>", false)
		return append(append(start, content...), end...), nil

	case "response":
		start, _ := b.tokenizer.Encode("<|tool_response>", false)
		content, _ := b.tokenizer.EncodeWithAllowed(string(comp.Data), false, []string{})
		end, _ := b.tokenizer.Encode("<tool_response|>", false)
		return append(append(start, content...), end...), nil

	default:
		return nil, fmt.Errorf("unsupported component type: %s", comp.CType)
	}
}
