package gemma4

import (
	"fmt"
	"strings"
	"testing"
)

func TestPrintGemma4StopTokens(t *testing.T) {
	ggufPath, err := findGemma4GGUF()
	if err != nil {
		t.Skipf("skipping: %v", err)
	}
	fmt.Printf("Using GGUF: %s\n", ggufPath)

	tok := loadGemma4Tokenizer(t, ggufPath)
	vocab := tok.Vocabulary()

	fmt.Println("\n--- Gemma 4 Stop & Control Tokens ---")

	// Traditional EOS/EOT
	fmt.Printf("BOS IDs: %v\n", vocab.BOS)
	fmt.Printf("EOS IDs: %v\n", vocab.EOS)

	// Known stop strings from renderers and parsers
	stopStrings := []string{
		"<eos>",
		"<end_of_turn>",
		"<turn|>",
		"<channel|>",
		"<tool_call|>",
		"<tool_response|>",
		"<tool|>",
		"<image|>",
		"<audio|>",
		"<|turn>",
		"<|channel>",
		"<|tool_call>",
		"<|tool_response>",
		"<|tool>",
		"<|image>",
		"<|audio>",
		"<|\"|>",
	}

	fmt.Println("\n--- Gemma 4 Encoding Comparison ---")
	fmt.Printf("%-20s | %-22s | %-22s\n", "String", "addSpecial=false", "addSpecial=true")
	fmt.Println(strings.Repeat("-", 70))

	// Configure vocab to add BOS/EOS for the "true" test
	tok.Vocabulary().AddBOS = true
	tok.Vocabulary().AddEOS = true

	for _, s := range stopStrings {
		idsFalse, _ := tok.Encode(s, false)
		idsTrue, _ := tok.Encode(s, true)
		fmt.Printf("%-20s | %-22v | %-22v\n", s, idsFalse, idsTrue)
	}

	// Also show a case where user types a "fake" token
	fakeToken := "<turn|> user prompt"
	idsFalse, _ := tok.Encode(fakeToken, false)
	idsTrue, _ := tok.Encode(fakeToken, true)
	fmt.Printf("%-20s | %-22v | %-22v\n", "Fake Injection", idsFalse, idsTrue)

	fmt.Println("\n--- Google Docs Stop Token Verification ---")
	docsStopTokens := map[int32]string{
		1:      "<eos>",
		106:    "<turn|>",
		50:     "<|tool_response>",
		258882: "<image|>",
		258883: "<audio|>",
	}

	for id, expected := range docsStopTokens {
		if int(id) < len(vocab.Values) {
			actual := vocab.Values[id]
			fmt.Printf("ID %-6d: Expected [%-16s] | Actual [%-16s]\n", id, expected, actual)
		} else {
			fmt.Printf("ID %-6d: Expected [%-16s] | Actual [OUT OF RANGE]\n", id, expected)
		}
	}

	fmt.Println("\n--- ACTIVE Stop Tokens in this Model ---")
    
    // Check if 50 is in the vocab as a control token
    for i, v := range vocab.Values {
        if i == 50 || i == 101 || i == 49 {
            fmt.Printf("Potential Stop ID %-6d: %s (Type: %d)\n", i, v, vocab.Types[i])
        }
    }
}
