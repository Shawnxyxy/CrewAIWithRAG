package vector

import (
	"testing"

	"crew_ai_with_rag/internal/config"
)

func TestSearchInMilvus(t *testing.T) {
	// 测试用配置
	embCfg := &config.EmbeddingConfig{
		APIType:   "openai",
		BaseURL:   "http://localhost:3000/v1",
		APIKey:    "sk-6kxLB9kLB7VW8jOxAe3aB261136e44989a31F19392162c4e",
		ModelName: "text-embedding-v1", // 阿里的embedding模型名
	}
	vector, err := GetEmbedding(embCfg, "健康档案检索测试")
	if err != nil {
		t.Fatalf("GetEmbedding failed: %v", err)
	}
	embCfg.Dimension = len(vector)
	t.Logf("Embedding dimension: %d", embCfg.Dimension)
	// 1. 维度校验
	if len(vector) != embCfg.Dimension {
		t.Fatalf("unexpected embedding dimension: %d", len(vector))
	}
	// 2. 非零校验
	hasNonZero := false
	for _, v := range vector {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Fatal("embedding vector is all zeros")
	}

	t.Logf("embedding generated, dim=%d", len(vector))
}