package tokenizer

import (
	"slices"
	"testing"
)

// mockVocabulary creates a simple vocab for testing
func mockVocabulary() *Vocabulary {
	v := &Vocabulary{
		Values: []string{"<pad>", "<eos>", "<bos>", "hello", " world", "<|image|>", "<turn|>", "<|think|>", "▁", "▁hello"},
		Types:  []int32{TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL, TOKEN_TYPE_NORMAL},
		Scores: make([]float32, 10),
	}
	return v
}

func TestSecureTokenizationCore(t *testing.T) {
	v := mockVocabulary()
	
	// Test both BPE and SentencePiece
	tokenizers := []struct {
		name string
		tok  Tokenizer
	}{
		{"BPE", NewBytePairEncoding(v)},
		{"SentencePiece", NewSentencePiece(v)},
	}

	for _, tc := range tokenizers {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Test Strict Neutralization (Allowed: [])
			ids, _ := tc.tok.EncodeWithAllowed("hello <eos>", false, []string{})
			for _, id := range ids {
				if v.Types[id] == TOKEN_TYPE_CONTROL {
					t.Errorf("[%s] Found control token %d in neutralized string", tc.name, id)
				}
			}

			// 2. Test Selective Whitelist (Allowed: [<|image|>])
			// We expect <|image|> to be ID 5, but <turn|> to be neutralized
			ids, _ = tc.tok.EncodeWithAllowed("<|image|><turn|>", false, []string{"<|image|>"})
			
			foundImage := false
			foundTurn := false
			for _, id := range ids {
				val := v.Values[id]
				if val == "<|image|>" {
					foundImage = true
				}
				if val == "<turn|>" {
					foundTurn = true
				}
			}
			
			if !foundImage {
				t.Errorf("[%s] Failed to match whitelisted <|image|>", tc.name)
			}
			if foundTurn {
				t.Errorf("[%s] Failed to neutralize restricted <turn|>", tc.name)
			}

			// 3. Test Space Normalization Loophole
			// SentencePiece often normalizes " <eos>" to "▁<eos>" (if ▁ exists) or handles it in Encode(text)
			// We want to ensure it doesn't resolve to ID 1
			ids, _ = tc.tok.EncodeWithAllowed(" <eos>", false, []string{})
			for _, id := range ids {
				if id == 1 {
					t.Errorf("[%s] Loophole found: ' <eos>' resolved to control ID 1", tc.name)
				}
			}

			// 4. Test Single String Exact Match
			ids, _ = tc.tok.EncodeWithAllowed("<|think|>", false, []string{})
			if len(ids) == 1 && ids[0] == 7 {
				t.Errorf("[%s] Loophole found: exact match '<|think|>' resolved to control ID 7", tc.name)
			}
		})
	}
}

func TestEncodeWithAllowedCompatibility(t *testing.T) {
	v := mockVocabulary()
	tok := NewBytePairEncoding(v)

	// nil allowed should behave like standard Encode
	idsStandard, _ := tok.Encode("<eos>", false)
	idsNil, _ := tok.EncodeWithAllowed("<eos>", false, nil)

	if !slices.Equal(idsStandard, idsNil) {
		t.Errorf("EncodeWithAllowed(nil) does not match Encode()")
	}
}
