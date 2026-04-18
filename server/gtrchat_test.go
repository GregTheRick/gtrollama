package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/tokenizer"
)

// MockTokenizer implements tokenizer.Tokenizer for testing
type MockTokenizer struct {
	specials map[string]int32
}

func NewMockTokenizer() *MockTokenizer {
	return &MockTokenizer{
		specials: map[string]int32{
			"<|turn>":    2,
			"<turn|>":    106,
			"<|channel>": 46,
			"<channel|>": 47,
			"<|image|>":  258880,
			"<|image>":   258880,
			"<image|>":   258882,
		},
	}
}

func (m *MockTokenizer) Encode(s string, addSpecial bool) ([]int32, error) {
	// Much smarter mock that recognizes structural tokens
	var res []int32

	remaining := s
	for len(remaining) > 0 {
		found := false
		// Try to match special tokens first
		for spec, id := range m.specials {
			if strings.HasPrefix(remaining, spec) {
				res = append(res, id)
				remaining = remaining[len(spec):]
				found = true
				break
			}
		}

		if !found {
			// Fallback to char-by-char for mock visibility
			r := rune(remaining[0])
			res = append(res, int32(r))
			remaining = remaining[1:]
		}
	}
	return res, nil
}

func (m *MockTokenizer) EncodeWithAllowed(s string, addSpecial bool, allowed []string) ([]int32, error) {
	// In the real Gemma4 tokenizer, restricted encoding means special tokens
	// are NOT matched even if they appear in the string.
	// Our mock should reflect this: if allowed is not nil, only match those.

	if allowed == nil {
		return m.Encode(s, addSpecial)
	}

	var res []int32
	remaining := s
	for len(remaining) > 0 {
		found := false
		for _, spec := range allowed {
			if id, ok := m.specials[spec]; ok && strings.HasPrefix(remaining, spec) {
				res = append(res, id)
				remaining = remaining[len(spec):]
				found = true
				break
			}
		}
		if !found {
			r := rune(remaining[0])
			res = append(res, int32(r))
			remaining = remaining[1:]
		}
	}
	return res, nil
}

func (m *MockTokenizer) Decode(tokens []int32) (string, error) {
	var sb strings.Builder
	for _, t := range tokens {
		matched := false
		for spec, id := range m.specials {
			if t == id {
				sb.WriteString(spec)
				matched = true
				break
			}
		}
		if !matched {
			sb.WriteString(fmt.Sprintf("[%d]", t))
		}
	}
	return sb.String(), nil
}

func (m *MockTokenizer) Vocabulary() *tokenizer.Vocabulary {
	values := make([]string, 300000)
	for s, id := range m.specials {
		values[id] = s
	}
	return &tokenizer.Vocabulary{Values: values}
}

func (m *MockTokenizer) Is(id int32, special tokenizer.Special) bool { return false }

func TestGTRPromptBuilder(t *testing.T) {
	builder := &GTRPromptBuilder{
		tokenizer: NewMockTokenizer(),
	}

	turns := []api.GTRChatTurn{
		{
			Role: "user",
			Components: []api.GTRChatComponent{
				{CType: "answer", Data: []byte(`{"text": "Hello"}`)},
			},
		},
	}

	tokens, images, err := builder.Build(context.Background(), turns)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Expecting: <|turn>user\nHello<turn|><|turn>model\n
	expected := "<|turn>user\nHello<turn|><|turn>model\n"
	detokenized, _ := builder.tokenizer.Decode(intSliceTo32(tokens))

	if detokenized != expected {
		t.Errorf("expected %q, got %q", expected, detokenized)
	}
	if len(images) != 0 {
		t.Errorf("expected 0 images, got %d", len(images))
	}
}

func TestGTRPromptBuilder_ModelTurn(t *testing.T) {
	builder := &GTRPromptBuilder{
		tokenizer: NewMockTokenizer(),
	}

	turns := []api.GTRChatTurn{
		{
			Role: "user",
			Components: []api.GTRChatComponent{
				{CType: "answer", Data: []byte(`{"text": "Hello"}`)},
			},
		},
		{
			Role: "model",
			Components: []api.GTRChatComponent{
				{CType: "thinking", Data: []byte(`{"text": "I am thinking"}`)},
			},
		},
	}

	tokens, _, err := builder.Build(context.Background(), turns)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Expecting: <|turn>user\nHello<turn|><|turn>model\n<|channel>thought\nI am thinking<channel|>
	// No trailing <turn|>
	expected := "<|turn>user\nHello<turn|><|turn>model\n<|channel>thought\nI am thinking<channel|>"
	detokenized, _ := builder.tokenizer.Decode(intSliceTo32(tokens))

	if detokenized != expected {
		t.Errorf("expected %q, got %q", expected, detokenized)
	}
}

// MockLlamaServer for testing Detokenize
type MockLlamaServer struct {
	llm.LlamaServer
	t         *testing.T
	tokenizer tokenizer.Tokenizer
}

func (m *MockLlamaServer) Detokenize(ctx context.Context, ids []int) (string, error) {
	return m.tokenizer.Decode(intSliceTo32(ids))
}

func TestGTRPromptBuilder_TokenLevelSafety(t *testing.T) {
	builder := &GTRPromptBuilder{
		tokenizer: NewMockTokenizer(),
	}

	turns := []api.GTRChatTurn{
		{
			Role: "user",
			Components: []api.GTRChatComponent{
				// User literally types <|image|>
				{CType: "answer", Data: []byte(`{"text": "<|image|>"}`)},
				// Real image component follows
				{CType: "image", Data: []byte(`{"multimodal": "YmFzZTY0ZGF0YQ=="}`)}, // "base64data"
			},
		},
	}

	tokens, images, err := builder.Build(context.Background(), turns)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Verify that we have exactly one structural image token (258880)
	// and that the user's <|image|> is represented as something else (text pieces)
	imgTokenCount := 0
	for _, t := range tokens {
		if t == 258880 {
			imgTokenCount++
		}
	}

	if imgTokenCount != 1 {
		t.Errorf("expected 1 structural image token (258880), got %d", imgTokenCount)
	}

	if len(images) != 1 {
		t.Errorf("expected 1 image in collector, got %d", len(images))
	}

	if images[0].ID != 0 {
		t.Errorf("expected image ID 0, got %d", images[0].ID)
	}
}

func TestGTRPromptBuilder_VisualVerification(t *testing.T) {
	var tok tokenizer.Tokenizer = NewMockTokenizer()

	// Optional: Attempt to load real Gemma4 tokenizer if available
	ggufPath, err := findGemma4GGUFForTest()
	if err == nil {
		realTok, err := loadGemma4TokenizerForTest(ggufPath)
		if err == nil {
			tok = realTok
			t.Logf("Using REAL Gemma 4 tokenizer from: %s", ggufPath)
		}
	}

	if _, ok := tok.(*MockTokenizer); ok {
		t.Log("Using enhanced MockTokenizer (no Gemma 4 model found)")
	}

	builder := &GTRPromptBuilder{
		tokenizer: tok,
	}

	turns := []api.GTRChatTurn{
		{
			Role: "system",
			Components: []api.GTRChatComponent{
				{CType: "systemtext", Data: []byte(`{"text": "You are a helpful assistant."}`)},
				{CType: "tool", Data: []byte(`[
					{
						"type": "function",
						"function": {
							"name": "get_weather",
							"description": "Get current weather for a location",
							"parameters": {
								"type": "object",
								"properties": {
									"location": { "type": "string" }
								}
							}
						}
					},
					{
						"type": "function",
						"function": {
							"name": "search",
							"description": "Search the web for information",
							"parameters": {
								"type": "object",
								"properties": {
									"query": { "type": "string" }
								}
							}
						}
					}
				]`)},
			},
		},
		{
			Role: "user",
			Components: []api.GTRChatComponent{
				{CType: "answer", Data: []byte(`{"text": "What is in this image? <|image|>"}`)},
				{CType: "image", Data: []byte(`{"multimodal": "YmFzZTY0ZGF0YQ=="}`)},
			},
		},
		{
			Role: "model",
			Components: []api.GTRChatComponent{
				{CType: "thinking", Data: []byte(`{"text": "I see a blue ball."}`)},
				{CType: "answer", Data: []byte(`{"text": "It is a blue ball."}`)},
			},
		},
	}

	tokens, _, err := builder.Build(context.Background(), turns)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	mockRunner := &MockLlamaServer{t: t, tokenizer: tok}
	text, _ := builder.SafeDetokenize(context.Background(), mockRunner, tokens)

	t.Logf("\n--- Detokenized Prompt ---\n%s\n", text)

	t.Logf("\n--- Token Stream breakdown ---\n")
	for i, tok := range tokens {
		piece, _ := builder.tokenizer.Decode([]int32{int32(tok)})
		t.Logf("%4d: %6d -> %q", i, tok, piece)
	}
}

func findGemma4GGUFForTest() (string, error) {
	modelsDir := envconfig.Models()
	manifestDir := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "gemma4")
	entries, err := os.ReadDir(manifestDir)
	if err != nil {
		return "", err
	}
	blobDir := filepath.Join(modelsDir, "blobs")
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(manifestDir, entry.Name()))
		if err != nil {
			continue
		}
		var manifest struct {
			Layers []struct {
				MediaType string `json:"mediaType"`
				Digest    string `json:"digest"`
			} `json:"layers"`
		}
		if err := json.Unmarshal(data, &manifest); err != nil {
			continue
		}
		for _, layer := range manifest.Layers {
			if layer.MediaType == "application/vnd.ollama.image.model" {
				blobPath := filepath.Join(blobDir, strings.Replace(layer.Digest, ":", "-", 1))
				if _, err := os.Stat(blobPath); err == nil {
					return blobPath, nil
				}
			}
		}
	}
	return "", fmt.Errorf("no gemma4 blob found")
}

func loadGemma4TokenizerForTest(ggufPath string) (tokenizer.Tokenizer, error) {
	f, err := gguf.Open(ggufPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	tokens := f.KeyValue("tokenizer.ggml.tokens").Strings()
	scores64 := f.KeyValue("tokenizer.ggml.scores").Floats()
	scores := make([]float32, len(scores64))
	for i, s := range scores64 {
		scores[i] = float32(s)
	}
	types64 := f.KeyValue("tokenizer.ggml.token_type").Ints()
	types := make([]int32, len(types64))
	for i, tt := range types64 {
		types[i] = int32(tt)
	}
	merges := f.KeyValue("tokenizer.ggml.merges").Strings()

	vocab := &tokenizer.Vocabulary{
		Values: tokens,
		Types:  types,
		Scores: scores,
		Merges: merges,
		BOS:    []int32{2},
		EOS:    []int32{1},
		AddBOS: false,
	}
	return tokenizer.NewBytePairEncodingWithOptions(vocab, []string{}, tokenizer.WithSentencePieceNormalizer()), nil
}

func intSliceTo32(s []int) []int32 {
	res := make([]int32, len(s))
	for i, v := range s {
		res[i] = int32(v)
	}
	return res
}
