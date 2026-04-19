package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/tokenizer"
	typemodel "github.com/ollama/ollama/types/model"
)

var AGENTIC_STRUCTURAL = []string{
	"<|tool>", "<tool|>",
	"<|tool_call>", "<tool_call|>",
	"<|tool_response>", "<tool_response|>",
	"<|\"|>",
}

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
			tokenizer:   builder.tokenizer,
			delimiterID: -1,
		}

		// Initialize Control Token IDs using the vocabulary's direct encoding
		// This ensures we get the atomic single ID for special symbols if available,
		// avoiding subword fragmentation.
		vocab := builder.tokenizer.Vocabulary()
		parser.thoughtStartIDs, _ = builder.tokenizer.Encode("<|channel>thought\n", false)
		parser.thoughtEndIDs, _ = builder.tokenizer.Encode("<channel|>", false)
		parser.toolStartIDs, _ = builder.tokenizer.Encode("<|tool_call>", false)
		parser.toolEndIDs, _ = builder.tokenizer.Encode("<tool_call|>", false)
		parser.delimiterID = vocab.Encode("<|\"|>")

		// If the direct vocabulary lookup for tool/thought start symbols
		// returns a valid ID, we prioritize that atomic ID.
		if id := vocab.Encode("<|channel>thought\n"); id != -1 {
			parser.thoughtStartIDs = []int32{id}
		}
		if id := vocab.Encode("<channel|>"); id != -1 {
			parser.thoughtEndIDs = []int32{id}
		}
		if id := vocab.Encode("<|tool_call>"); id != -1 {
			parser.toolStartIDs = []int32{id}
		}
		if id := vocab.Encode("<tool_call|>"); id != -1 {
			parser.toolEndIDs = []int32{id}
		}

		slog.Debug("GTR Parser initialized",
			"thought_start", parser.thoughtStartIDs,
			"tool_start", parser.toolStartIDs,
			"delimiter", parser.delimiterID)

		if opts.Stop == nil {
			opts.Stop = []string{}
		}
		opts.Stop = append(opts.Stop, "<turn|>", "<eos>", "<|tool_response>")

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
	tokenizer  tokenizer.Tokenizer

	state       string // "", "thinking", "tool_call"
	tokenBuffer []int32

	// Control Token IDs
	thoughtStartIDs []int32 // <|channel>thought\n
	thoughtEndIDs   []int32 // <channel|>
	toolStartIDs    []int32 // <|tool_call>
	toolEndIDs      []int32 // <tool_call|>
	delimiterID     int32   // <|"|>
	hadToolCall     bool
}

func (p *GTRResponseParser) Parse(cr llm.CompletionResponse) {
	if p.streamMode == "raw" {
		p.onEvent(api.GTRChatResponseEvent{Type: "text", Content: cr.Content})
		if cr.Done {
			p.onEvent(api.GTRChatResponseEvent{Type: "done", Status: "complete"})
		}
		return
	}

	if len(cr.TokenIDs) > 0 {
		slog.Debug("GTR Parser incoming tokens", "state", p.state, "ids", cr.TokenIDs, "buf_len", len(p.tokenBuffer))
	}
	for _, id := range cr.TokenIDs {
		p.tokenBuffer = append(p.tokenBuffer, int32(id))
	}

	for {
		found := false

		if p.state == "thinking" {
			if idx := findSequence(p.tokenBuffer, p.thoughtEndIDs); idx != -1 {
				if idx > 0 {
					p.emitTokens(p.tokenBuffer[:idx])
				}
				p.state = ""
				p.tokenBuffer = p.tokenBuffer[idx+len(p.thoughtEndIDs):]
				found = true
			}
		} else if p.state == "tool_call" {
			if idx := findSequence(p.tokenBuffer, p.toolEndIDs); idx != -1 {
				p.emitToolCall(p.tokenBuffer[:idx])
				p.hadToolCall = true
				p.state = ""
				p.tokenBuffer = p.tokenBuffer[idx+len(p.toolEndIDs):]
				found = true
			}
		} else {
			// Ground state: check for transitions or delimiters
			if idx := findSequence(p.tokenBuffer, p.thoughtStartIDs); idx != -1 {
				if idx > 0 {
					p.emitTokens(p.tokenBuffer[:idx])
				}
				p.state = "thinking"
				p.tokenBuffer = p.tokenBuffer[idx+len(p.thoughtStartIDs):]
				found = true
			} else if idx := findSequence(p.tokenBuffer, p.toolStartIDs); idx != -1 {
				if idx > 0 {
					p.emitTokens(p.tokenBuffer[:idx])
				}
				p.state = "tool_call"
				p.tokenBuffer = p.tokenBuffer[idx+len(p.toolStartIDs):]
				found = true
			} else if p.delimiterID != -1 {
				if idx := findID(p.tokenBuffer, p.delimiterID); idx != -1 {
					if idx > 0 {
						p.emitTokens(p.tokenBuffer[:idx])
					}
					p.tokenBuffer = p.tokenBuffer[idx+1:]
					found = true
				}
			}
		}

		if !found {
			break
		}
	}

	// Safety emission for text/thought
	if p.state != "tool_call" && len(p.tokenBuffer) > 5 {
		safeLen := len(p.tokenBuffer) - 5
		p.emitTokens(p.tokenBuffer[:safeLen])
		p.tokenBuffer = p.tokenBuffer[safeLen:]
	}

	if cr.Done {
		if len(p.tokenBuffer) > 0 && p.state != "tool_call" {
			p.emitTokens(p.tokenBuffer)
		}
		status := "complete"
		if p.hadToolCall {
			status = "call_wait"
		}
		p.onEvent(api.GTRChatResponseEvent{Type: "done", Status: status})
	}
}

func findSequence(buf, seq []int32) int {
	if len(seq) == 0 {
		return -1
	}
	for i := 0; i <= len(buf)-len(seq); i++ {
		match := true
		for j := 0; j < len(seq); j++ {
			if buf[i+j] != seq[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

func findID(buf []int32, id int32) int {
	for i, val := range buf {
		if val == id {
			return i
		}
	}
	return -1
}

func (p *GTRResponseParser) emitTokens(tokens []int32) {
	if len(tokens) == 0 {
		return
	}
	content, _ := p.tokenizer.Decode(tokens)
	p.emit(content)
}

func (p *GTRResponseParser) emitToolCall(tokens []int32) {
	content, _ := p.tokenizer.Decode(tokens)
	slog.Debug("GTR Parser decoding tool call", "raw", content)
	callData := parseGemma4ToolCall(content)
	p.onEvent(api.GTRChatResponseEvent{
		Type:     "tool_call",
		ToolCall: callData,
	})
}

func (p *GTRResponseParser) emit(content string) {
	if content == "" || p.state == "tool_call" {
		return
	}
	contentType := "text"
	switch p.state {
	case "thinking":
		contentType = "thinking"
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

	// Gemma 4 IT models require a BOS token at the beginning of the prompt
	fullTokens = append(fullTokens, 2)

	// Check if thinking mode is requested anywhere in the conversation
	thinking := false
	for _, t := range turns {
		if t.ThinkingEnabled {
			thinking = true
			break
		}
	}

	// Trigger Thinking Mode if requested.
	// Per official spec, <|think|> must be at the start of the system prompt.
	if thinking {
		t1, _ := b.tokenizer.Encode("<|turn>system\n<|think|><turn|>\n", false)
		fullTokens = append(fullTokens, t1...)
	}

	for i, turn := range turns {
		// Turn Start: <|turn>role\n
		// To prevent shattering, we encode the structural marker atomically
		t1, err := b.tokenizer.Encode("<|turn>", false)
		if err != nil {
			return nil, nil, err
		}
		t2, err := b.tokenizer.Encode(turn.Role+"\n", false)
		if err != nil {
			return nil, nil, err
		}
		fullTokens = append(fullTokens, append(t1, t2...)...)

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
			tokens, err := b.tokenizer.Encode("<turn|>\n", false)
			if err != nil {
				return nil, nil, err
			}
			fullTokens = append(fullTokens, tokens...)
		}
	}

	// Ensure model turn start if the last turn was not a model turn
	if len(turns) > 0 && turns[len(turns)-1].Role != "model" {
		t1, _ := b.tokenizer.Encode("<|turn>", false)
		t2, _ := b.tokenizer.Encode("model\n", false)
		fullTokens = append(fullTokens, append(t1, t2...)...)
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
	case "system_text", "answer", "thinking":
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

	case "toolcall", "tool_call":
		var data api.GTRToolCallData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}

		// Structure: <|tool_call>call:name{key:<|"|>val<|"|>,...}<tool_call|>
		t1, _ := b.tokenizer.Encode("<|tool_call>call:", false)
		t2, _ := b.tokenizer.EncodeWithAllowed(data.Name, false, []string{}) // Safe tool name
		t3, _ := b.tokenizer.Encode("{", false)
		tokens := append(append(t1, t2...), t3...)

		qTok, _ := b.tokenizer.Encode("<|\"|>", false)

		for i, arg := range data.Args {
			if i > 0 {
				sep, _ := b.tokenizer.Encode(",", false)
				tokens = append(tokens, sep...)
			}
			k, _ := b.tokenizer.EncodeWithAllowed(arg.Key, false, []string{}) // Safe key
			c, _ := b.tokenizer.Encode(":", false)
			tokens = append(append(tokens, k...), c...)

			v, _ := b.tokenizer.EncodeWithAllowed(arg.Val, false, []string{}) // Safe value
			tokens = append(append(append(tokens, qTok...), v...), qTok...)
		}

		closeBrace, _ := b.tokenizer.Encode("}", false)
		tEnd, _ := b.tokenizer.Encode("<tool_call|>", false)
		return append(append(tokens, closeBrace...), tEnd...), nil

	case "toolresponse", "tool_response":
		var data api.GTRToolCallData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}

		// Structure: <|tool_response>response:name{key:<|"|>val<|"|>,...}<tool_response|>
		t1, _ := b.tokenizer.Encode("<|tool_response>response:", false)
		t2, _ := b.tokenizer.EncodeWithAllowed(data.Name, false, []string{})
		t3, _ := b.tokenizer.Encode("{", false)
		tokens := append(append(t1, t2...), t3...)

		qTok, _ := b.tokenizer.Encode("<|\"|>", false)

		for i, arg := range data.Args {
			if i > 0 {
				sep, _ := b.tokenizer.Encode(",", false)
				tokens = append(tokens, sep...)
			}
			k, _ := b.tokenizer.EncodeWithAllowed(arg.Key, false, []string{})
			c, _ := b.tokenizer.Encode(":", false)
			tokens = append(append(tokens, k...), c...)

			// Assume all the response results in our demo are strings for now,
			// though the spec allows numbers etc.
			v, _ := b.tokenizer.EncodeWithAllowed(arg.Val, false, []string{})
			tokens = append(append(append(tokens, qTok...), v...), qTok...)
		}

		tEnd, _ := b.tokenizer.Encode("}<tool_response|>", false)
		return append(tokens, tEnd...), nil

	case "image":
		var data api.GTRMultimodalData
		if err := json.Unmarshal(comp.Data, &data); err != nil {
			return nil, err
		}
		imgData, err := base64.StdEncoding.DecodeString(data.Multimodal)
		if err != nil {
			return nil, fmt.Errorf("failed to decode image base64: %v", err)
		}
		b.images = append(b.images, llm.ImageData{Data: imgData})
		return []int32{258880}, nil

	case "toolschema", "tool_schema":
		var wrapper struct {
			Tools []gemmaTool `json:"tools"`
		}
		if err := json.Unmarshal(comp.Data, &wrapper); err != nil {
			return nil, err
		}

		var tokens []int32
		for i, t := range wrapper.Tools {
			if i > 0 {
				nl, _ := b.tokenizer.Encode("\n", false)
				tokens = append(tokens, nl...)
			}
			tokens = append(tokens, formatGemmaFunctionDeclarationTokens(b.tokenizer, t)...)
		}

		start, _ := b.tokenizer.Encode("<|tool>", false)
		end, _ := b.tokenizer.Encode("<tool|>", false)
		return append(append(start, tokens...), end...), nil

	default:
		return nil, fmt.Errorf("unsupported component type: %s", comp.CType)
	}
}

type gemmaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

func formatGemmaFunctionDeclarationTokens(tok tokenizer.Tokenizer, tool gemmaTool) []int32 {
	var tokens []int32

	// declaration:name{
	t, _ := tok.Encode("declaration:", false)
	tokens = append(tokens, t...)
	t, _ = tok.EncodeWithAllowed(tool.Function.Name, false, []string{})
	tokens = append(tokens, t...)
	t, _ = tok.Encode("{\n", false)
	tokens = append(tokens, t...)

	qTok, _ := tok.Encode("<|\"|>", false)

	if tool.Function.Description != "" {
		t, _ = tok.Encode("    description:", false)
		tokens = append(tokens, t...)
		tokens = append(tokens, qTok...)
		t, _ = tok.EncodeWithAllowed(tool.Function.Description, false, []string{})
		tokens = append(tokens, t...)
		tokens = append(tokens, qTok...)
		t, _ = tok.Encode(",\n", false)
		tokens = append(tokens, t...)
	}

	params := tool.Function.Parameters
	if params != nil {
		t, _ = tok.Encode("    parameters:{\n", false)
		tokens = append(tokens, t...)

		if props, ok := params["properties"].(map[string]interface{}); ok {
			var nReq []string
			if nr, ok := params["required"].([]interface{}); ok {
				for _, r := range nr {
					nReq = append(nReq, fmt.Sprint(r))
				}
			}
			t, _ = tok.Encode("        properties:{\n", false)
			tokens = append(tokens, t...)
			tokens = append(tokens, formatGemmaParametersTokens(tok, props, nReq, 2)...)
			t, _ = tok.Encode("\n        }", false)
			tokens = append(tokens, t...)
		}

		if req, ok := params["required"].([]interface{}); ok && len(req) > 0 {
			t, _ = tok.Encode("},required:[", false)
			tokens = append(tokens, t...)
			for i, r := range req {
				if i > 0 {
					t, _ = tok.Encode(",", false)
					tokens = append(tokens, t...)
				}
				tokens = append(tokens, qTok...)
				t, _ = tok.EncodeWithAllowed(fmt.Sprint(r), false, []string{})
				tokens = append(tokens, t...)
				tokens = append(tokens, qTok...)
			}
			t, _ = tok.Encode("],type:<|\"|>OBJECT<|\"|> }", false)
			tokens = append(tokens, t...)
		} else {
			t, _ = tok.Encode("},type:<|\"|>OBJECT<|\"|> }", false)
			tokens = append(tokens, t...)
		}

		t, _ = tok.Encode("\n    }", false)
		tokens = append(tokens, t...)
	}

	t, _ = tok.Encode("\n}", false)
	return append(tokens, t...)
}

func formatGemmaParametersTokens(tok tokenizer.Tokenizer, props map[string]interface{}, required []string, depth int) []int32 {
	indent := strings.Repeat("    ", depth+1)
	var tokens []int32

	keys := make([]string, 0, len(props))
	for k := range props {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	qTok, _ := tok.Encode("<|\"|>", false)

	for i, key := range keys {
		if i > 0 {
			t, _ := tok.Encode(",\n", false)
			tokens = append(tokens, t...)
		}

		val, ok := props[key].(map[string]interface{})
		if !ok {
			continue
		}

		t, _ := tok.Encode(indent, false)
		tokens = append(tokens, t...)
		t, _ = tok.EncodeWithAllowed(key, false, []string{})
		tokens = append(tokens, t...)
		t, _ = tok.Encode(":{", false)
		tokens = append(tokens, t...)

		first := true
		if desc, ok := val["description"].(string); ok {
			t, _ = tok.Encode("description:", false)
			tokens = append(tokens, t...)
			tokens = append(tokens, qTok...)
			t, _ = tok.EncodeWithAllowed(desc, false, []string{})
			tokens = append(tokens, t...)
			tokens = append(tokens, qTok...)
			first = false
		}

		typeStr := ""
		if t, ok := val["type"].(string); ok {
			typeStr = strings.ToUpper(t)
		}

		if typeStr == "OBJECT" {
			if !first {
				t, _ = tok.Encode(",", false)
				tokens = append(tokens, t...)
			}
			if nested, ok := val["properties"].(map[string]interface{}); ok {
				var nReq []string
				if nr, ok := val["required"].([]interface{}); ok {
					for _, r := range nr {
						nReq = append(nReq, fmt.Sprint(r))
					}
				}
				t, _ = tok.Encode("properties:{\n", false)
				tokens = append(tokens, t...)
				tokens = append(tokens, formatGemmaParametersTokens(tok, nested, nReq, depth+1)...)
				t, _ = tok.Encode(fmt.Sprintf("\n%s}", indent), false)
				tokens = append(tokens, t...)
				first = false
			}
		}

		if typeStr != "" {
			if !first {
				t, _ = tok.Encode(",", false)
				tokens = append(tokens, t...)
			}
			t, _ = tok.Encode("type:", false)
			tokens = append(tokens, t...)
			tokens = append(tokens, qTok...)
			t, _ = tok.EncodeWithAllowed(typeStr, false, []string{})
			tokens = append(tokens, t...)
			tokens = append(tokens, qTok...)
		}

		t, _ = tok.Encode("}", false)
		tokens = append(tokens, t...)
	}

	return tokens
}

func parseGemma4ToolCall(s string) *api.GTRToolCallData {
	res := &api.GTRToolCallData{}
	s = strings.TrimSpace(s)

	// Gemma4 usually output starts with "call:name{"
	// We'll look for the first colon and the first brace
	colonIdx := strings.Index(s, ":")
	braceIdx := strings.Index(s, "{")

	if colonIdx != -1 && (braceIdx == -1 || colonIdx < braceIdx) {
		// Verify if the prefix is indeed "call" (case-insensitive)
		prefix := strings.ToLower(strings.TrimSpace(s[:colonIdx]))
		if prefix == "call" {
			s = s[colonIdx+1:]
			braceIdx = strings.Index(s, "{")
		}
	}

	if braceIdx == -1 {
		res.Name = strings.TrimSpace(s)
		return res
	}

	res.Name = strings.TrimSpace(s[:braceIdx])
	argsStr := s[braceIdx+1:]
	if strings.HasSuffix(argsStr, "}") {
		argsStr = argsStr[:len(argsStr)-1]
	}

	// Split by comma
	parts := strings.Split(argsStr, ",")
	for _, p := range parts {
		kv := strings.SplitN(p, ":", 2)
		if len(kv) == 2 {
			k := strings.TrimSpace(kv[0])
			v := kv[1]
			// Clean up structural markers and quotes from value
			v = strings.ReplaceAll(v, "<|\"|>", "")
			v = strings.Trim(v, " \t\n\r\"'")
			res.Args = append(res.Args, api.GTRToolArg{Key: k, Val: v})
		}
	}

	return res
}
