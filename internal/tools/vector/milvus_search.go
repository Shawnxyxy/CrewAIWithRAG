package vector

import (
	"context"
	"fmt"

	"crew_ai_with_rag/internal/config"

	"github.com/sashabaranov/go-openai"
)

func SearchInMilvus(query string, topK int) (string, error) {
	embCfg, err := config.LoadEmbeddingConfig("internal/config/embedding.yaml")
	if err != nil {
		return "", fmt.Errorf("加载 embedding 配置失败: %v", err)
	}
	vector, err := GetEmbedding(embCfg, query)
	if err != nil {
		return "", fmt.Errorf("failed to get embedding: %v", err)
	}
	return fmt.Sprintf("生成向量成功，长度：%d", len(vector)), nil
}

func GetEmbedding(cfg *config.EmbeddingConfig, text string) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("text is empty")
	}
	oaCfg := openai.DefaultConfig(cfg.APIKey)
	oaCfg.BaseURL = cfg.BaseURL // 使用 one-api 的URL
	client := openai.NewClientWithConfig(oaCfg)
	resp, err := client.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{
			Input: []string{text},
			Model: openai.EmbeddingModel(cfg.ModelName),
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %v", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}
	return resp.Data[0].Embedding, nil
}