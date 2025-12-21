package crew

import (
	"testing"

	"crew_ai_with_rag/internal/config"
	"crew_ai_with_rag/internal/tools/vector"
)

var embCfg = &config.EmbeddingConfig{
	APIType:   "openai",
	BaseURL:   "http://localhost:3000/v1",
	APIKey:    "sk-6kxLB9kLB7VW8jOxAe3aB261136e44989a31F19392162c4e",
	ModelName: "text-embedding-v1", // 阿里的embedding模型名
}

func TestBuildContext(t *testing.T) {
	vs, err := vector.NewVectorStore("localhost:19530", embCfg, "crew_test_collection", "my_varchar", "my_vector", 3)
	if err != nil {
		t.Fatalf("NewVectorStore failed: %v", err)
	}
	if err := vs.EnsureCollection(1536); err != nil {
		t.Fatalf("EnsureCollection failed: %v", err)
	}
	texts := []string{"测试文本1", "测试文本2", "测试文本3"}
	if err := vs.InsertBatchTexts(texts); err != nil {
		t.Fatalf("InsertBatchTexts failed: %v", err)
	}
	crew := NewCrew(vs, embCfg)

	queries := []string{"测试文本1", "测试文本2"}
	results, err := crew.BuildContext(queries, 2)
	if err != nil {
		t.Fatalf("BuildContext failed: %v", err)
	}
	if len(results) != len(queries) {
		t.Fatalf("expected %d results, got %d", len(queries), len(results))
	}
	for i, r := range results {
		t.Logf("Query %d context:\n%s", i, r)
	}
}