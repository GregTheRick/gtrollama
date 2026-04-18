package gemma4

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
)

func TestPrintGGUFStopMetadata(t *testing.T) {
	ggufPath, err := findGemma4GGUF()
	if err != nil {
		t.Skipf("skipping: %v", err)
	}

	f, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatalf("failed to open gguf: %v", err)
	}
	defer f.Close()

	fmt.Println("\n--- Raw GGUF Stop Token Metadata (Gemma 4) ---")
    
    // Check single EOS ID
    eosID := f.KeyValue("tokenizer.ggml.eos_token_id").Uint()
    fmt.Printf("tokenizer.ggml.eos_token_id: %d\n", eosID)

    // Check EOT ID if exists
    eotID := f.KeyValue("tokenizer.ggml.eot_token_id").Uint()
    if eotID != 0 {
        fmt.Printf("tokenizer.ggml.eot_token_id: %d\n", eotID)
    } else {
        fmt.Printf("tokenizer.ggml.eot_token_id: [NOT FOUND, defaulting to 106 in code]\n")
    }

    // Check EOS IDs array
    eosIDs := f.KeyValue("tokenizer.ggml.eos_token_ids").Uints()
    fmt.Printf("tokenizer.ggml.eos_token_ids: %v\n", eosIDs)

    // Verify IDs mentioned in docs
    docIDs := []uint64{1, 106, 50, 258882, 258883}
    for _, id := range docIDs {
        found := false
        if eosID == id || eotID == id {
            found = true
        } else {
            for _, eid := range eosIDs {
                if eid == id {
                    found = true
                    break
                }
            }
        }
        
        status := "MISSING from GGUF metadata"
        if found {
            status = "FOUND in GGUF metadata"
        }
        
        fmt.Printf("Doc ID %-6d: %s\n", id, status)
    }
}
