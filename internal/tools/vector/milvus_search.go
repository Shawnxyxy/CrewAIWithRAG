package vector

import (
	"context"
	"fmt"

	"crew_ai_with_rag/internal/config"

	"github.com/sashabaranov/go-openai"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func SearchInMilvus(query string, topK int) (string, error) {
	embCfg, err := config.LoadEmbeddingConfig("../../config/embedding.yaml")
	if err != nil {
		return "", fmt.Errorf("加载 embedding 配置失败: %v", err)
	}
	vector, err := GetEmbedding(embCfg, query)
	if err != nil {
		return "", fmt.Errorf("failed to get embedding: %v", err)
	}
	return fmt.Sprintf("生成向量成功，长度：%d", len(vector)), nil
}

// 单条向量生成,只处理一条文本，返回对应的向量
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

// 批量向量生成,处理多条文本，返回多条向量
func GenerateBatchEmbedding(cfg *config.EmbeddingConfig, text []string) ([]([]float32), error) {
	if len(text) == 0 {
		return nil, fmt.Errorf("text is empty")
	}
	var results [][]float32
	batchSize := 25

	for i := 0; i < len(text); i += batchSize {
		end := i + batchSize
		if end > len(text) {
			end = len(text)
		}
		batch := text[i:end]
		// 对当前批次生成向量
		for _, t := range batch {
			vec, err := GetEmbedding(cfg, t)
			if err != nil {
				return nil, fmt.Errorf("failed to get embedding for text '%s': %v", t, err)
			}
			results = append(results, vec)
		}
	}
	return results, nil
}

// 单向量搜索
func (mc *MilvusClient) SearchSingle(collectionName string, queryVector []float32, topK int) ([]milvusclient.ResultSet, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 封装查询向量
	vectors := []entity.Vector{entity.FloatVector(queryVector)}

	// 发起搜索
	resultsSets, err := mc.Client.Search(ctx, milvusclient.NewSearchOption(
		collectionName,
		topK,
		vectors,
	).WithANNSField("my_vector").WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return nil, fmt.Errorf("milvus search failed: %v", err)
	}
	// 遍历打印结果
	for _, resultSet := range resultsSets {
		fmt.Println("IDs: ", resultSet.IDs.FieldData().GetScalars())
		fmt.Println("Scores: ", resultSet.Scores)
	}
	return resultsSets, nil
}
// 批量向量搜索
func (mc *MilvusClient) SearchBatch(collectionName string, queryVector [][]float32, topK int) ([]milvusclient.ResultSet, error) {
	if len(queryVector) == 0 {
		return nil, fmt.Errorf("queryVector is empty")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// 封装每一条查询向量
	vector := make([]entity.Vector, 0, len(queryVector))
	for _, vec := range queryVector {
		vector = append(vector, entity.FloatVector(vec))
	}
	// 批量搜索
	resultSets, err := mc.Client.Search(ctx, milvusclient.NewSearchOption(
		collectionName,
		topK,
		vector,
	).WithANNSField("my_vector").WithConsistencyLevel(entity.ClStrong))
	if err != nil {
		return nil, fmt.Errorf("milvus batch search failed: %v", err)
	}

	for i, rs := range resultSets {
		fmt.Printf("Query %d IDs: %v\n", i, rs.IDs.FieldData().GetScalars())
		fmt.Printf("Query %d Scores: %v\n", i, rs.Scores)
	}
	return resultSets, nil
}