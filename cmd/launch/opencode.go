package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	modeltype "github.com/ollama/ollama/types/model"
)

// OpenCode implements Runner and Editor for OpenCode integration
type OpenCode struct{}

func (o *OpenCode) String() string { return "OpenCode" }

// findOpenCode returns the opencode binary path, checking PATH first then the
// curl installer location (~/.opencode/bin) which may not be on PATH yet.
func findOpenCode() (string, bool) {
	if p, err := exec.LookPath("opencode"); err == nil {
		return p, true
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", false
	}
	name := "opencode"
	if runtime.GOOS == "windows" {
		name = "opencode.exe"
	}
	fallback := filepath.Join(home, ".opencode", "bin", name)
	if _, err := os.Stat(fallback); err == nil {
		return fallback, true
	}
	return "", false
}

func (o *OpenCode) Run(model string, args []string) error {
	opencodePath, ok := findOpenCode()
	if !ok {
		return fmt.Errorf("opencode is not installed, install from https://opencode.ai")
	}

	cmd := exec.Command(opencodePath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (o *OpenCode) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	var paths []string
	p := filepath.Join(home, ".config", "opencode", "opencode.json")
	if _, err := os.Stat(p); err == nil {
		paths = append(paths, p)
	}
	sp := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if _, err := os.Stat(sp); err == nil {
		paths = append(paths, sp)
	}
	return paths
}

func (o *OpenCode) Edit(modelList []string) error {
	if len(modelList) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".config", "opencode", "opencode.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config) // Ignore parse errors; treat missing/corrupt files as empty
	}

	config["$schema"] = "https://opencode.ai/config.json"

	provider, ok := config["provider"].(map[string]any)
	if !ok {
		provider = make(map[string]any)
	}

	ollama, ok := provider["ollama"].(map[string]any)
	if !ok {
		ollama = map[string]any{
			"npm":  "@ai-sdk/openai-compatible",
			"name": "Ollama",
			"options": map[string]any{
				"baseURL": envconfig.Host().String() + "/v1",
			},
		}
	}

	// Migrate legacy provider name
	if name, _ := ollama["name"].(string); name == "Ollama (local)" {
		ollama["name"] = "Ollama"
	}

	models, ok := ollama["models"].(map[string]any)
	if !ok {
		models = make(map[string]any)
	}

	selectedSet := make(map[string]bool)
	for _, m := range modelList {
		selectedSet[m] = true
	}

	for name, cfg := range models {
		if cfgMap, ok := cfg.(map[string]any); ok {
			if isOllamaModel(cfgMap) && !selectedSet[name] {
				delete(models, name)
			}
		}
	}

	client := api.NewClient(envconfig.Host(), http.DefaultClient)
	ctx := context.Background()

	for _, model := range modelList {
		if existing, ok := models[model].(map[string]any); ok {
			// migrate existing models without _launch marker
			if isOllamaModel(existing) {
				existing["_launch"] = true
				if name, ok := existing["name"].(string); ok {
					existing["name"] = strings.TrimSuffix(name, " [Ollama]")
				}
			}
			if isCloudModelName(model) {
				if l, ok := lookupCloudModelLimit(model); ok {
					existing["limit"] = map[string]any{
						"context": l.Context,
						"output":  l.Output,
					}
				}
			}
			// Update reasoning capability for managed models
			if isOllamaModel(existing) {
				applyOpenCodeReasoning(ctx, client, model, existing)
			}
			continue
		}
		entry := map[string]any{
			"name":    model,
			"_launch": true,
		}
		if isCloudModelName(model) {
			if l, ok := lookupCloudModelLimit(model); ok {
				entry["limit"] = map[string]any{
					"context": l.Context,
					"output":  l.Output,
				}
			}
		}
		applyOpenCodeReasoning(ctx, client, model, entry)
		models[model] = entry
	}

	ollama["models"] = models
	provider["ollama"] = ollama
	config["provider"] = provider
	config["model"] = "ollama/" + modelList[0]

	configData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(configPath, configData); err != nil {
		return err
	}

	statePath := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		return err
	}

	state := map[string]any{
		"recent":   []any{},
		"favorite": []any{},
		"variant":  map[string]any{},
	}
	if data, err := os.ReadFile(statePath); err == nil {
		_ = json.Unmarshal(data, &state) // Ignore parse errors; use defaults
	}

	recent, _ := state["recent"].([]any)

	modelSet := make(map[string]bool)
	for _, m := range modelList {
		modelSet[m] = true
	}

	// Filter out existing Ollama models we're about to re-add
	newRecent := slices.DeleteFunc(slices.Clone(recent), func(entry any) bool {
		e, ok := entry.(map[string]any)
		if !ok || e["providerID"] != "ollama" {
			return false
		}
		modelID, _ := e["modelID"].(string)
		return modelSet[modelID]
	})

	// Prepend models in reverse order so first model ends up first
	for _, model := range slices.Backward(modelList) {
		newRecent = slices.Insert(newRecent, 0, any(map[string]any{
			"providerID": "ollama",
			"modelID":    model,
		}))
	}

	const maxRecentModels = 10
	newRecent = newRecent[:min(len(newRecent), maxRecentModels)]

	state["recent"] = newRecent

	stateData, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(statePath, stateData)
}

func (o *OpenCode) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	config, err := fileutil.ReadJSON(filepath.Join(home, ".config", "opencode", "opencode.json"))
	if err != nil {
		return nil
	}
	provider, _ := config["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)
	if len(models) == 0 {
		return nil
	}
	keys := slices.Collect(maps.Keys(models))
	slices.Sort(keys)
	return keys
}

// applyOpenCodeReasoning detects thinking capability and sets reasoning config
// on the model entry. When the model supports thinking, it sets "reasoning": true
// and configures variants for the OpenCode TUI:
//   - GPT-OSS: supports variable effort levels (low/medium/high) and defaults to
//     medium via options. Thinking cannot be turned off.
//   - Other models: only support on/off. Disables built-in low/medium/high variants
//     and adds a "none" variant so users can toggle thinking off via Ctrl+T.
//
// When the model does not support thinking, it removes stale reasoning config.
func applyOpenCodeReasoning(ctx context.Context, client *api.Client, modelName string, entry map[string]any) {
	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return
	}

	if slices.Contains(resp.Capabilities, modeltype.CapabilityThinking) {
		entry["reasoning"] = true

		if strings.Contains(modelName, "gpt-oss") {
			// GPT-OSS models support variable thinking effort levels
			// and cannot turn thinking off. Keep the built-in
			// low/medium/high variants as-is and default to medium.
			options, ok := entry["options"].(map[string]any)
			if !ok {
				options = make(map[string]any)
			}
			options["reasoningEffort"] = "medium"
			entry["options"] = options
		} else {
			// Most models only support thinking on or off.
			// Disable the built-in low/medium/high variants and add none.
			entry["variants"] = map[string]any{
				"none":   map[string]any{"reasoningEffort": "none"},
				"low":    map[string]any{"disabled": true},
				"medium": map[string]any{"disabled": true},
				"high":   map[string]any{"disabled": true},
			}
		}
	} else {
		delete(entry, "reasoning")
		delete(entry, "variants")
		// Clean up options.reasoningEffort if it was previously set
		if options, ok := entry["options"].(map[string]any); ok {
			delete(options, "reasoningEffort")
			if len(options) == 0 {
				delete(entry, "options")
			}
		}
	}
}

// isOllamaModel reports whether a model config entry is managed by us
func isOllamaModel(cfg map[string]any) bool {
	if v, ok := cfg["_launch"].(bool); ok && v {
		return true
	}
	// previously used [Ollama] as a suffix for the model managed by ollama launch
	if name, ok := cfg["name"].(string); ok {
		return strings.HasSuffix(name, "[Ollama]")
	}
	return false
}
