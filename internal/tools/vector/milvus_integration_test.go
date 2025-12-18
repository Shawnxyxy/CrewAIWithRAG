package vector

import (
	"context"
	"testing"
	"time"

	"crew_ai_with_rag/internal/config"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
)

func TestMilvusFullFlow(t *testing.T) {
	// 基本配置
	milvusAddr := "localhost:19530"
	collectionName := "health_records_integration_test"

	// 加载 embedding 配置
	embCfg, err := config.LoadEmbeddingConfig("../../config/embedding.yaml")
	if err != nil {
		t.Fatalf("LoadEmbeddingConfig failed: %v", err)
	}

	// 创建客户端
	mc, err := NewMilvusClient(milvusAddr)
	if err != nil {
		t.Fatalf("Failed to create Milvus client: %v", err)
	}

	// 确保 collection 是干净的
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := EnsureFreshCollection(ctx, mc, collectionName); err != nil {
		t.Fatalf("EnsureFreshCollection failed: %v", err)
	}

	// 创建 collection
	if err := mc.CreateCollection(collectionName, 1536); err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	// 向量生成
	texts := []string{
		"健康档案检索测试",
		"用户健康状况分析",
		"今天的天气如何？",
	}
	vectorsBatch, err := GenerateBatchEmbedding(embCfg, texts)
	if err != nil {
		t.Fatalf("GenerateBatchEmbedding failed: %v", err)
	}

	// 插入数据
	ids := make([]int64, len(vectorsBatch))
	for i := range vectorsBatch {
		ids[i] = int64(i + 1)
	}
	insertOpt := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithInt64Column("my_id", ids).
		WithFloatVectorColumn("my_vector", len(vectorsBatch[0]), vectorsBatch).
		WithVarcharColumn("my_varchar", texts)
	if _, err := mc.Client.Insert(context.Background(), insertOpt); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// 创建索引
	indexTask, err := mc.Client.CreateIndex(context.Background(), milvusclient.NewCreateIndexOption(
		collectionName,
		"my_vector",
		index.NewAutoIndex(entity.COSINE)))
	if err != nil {
		t.Fatalf("CreateIndex failed: %v", err)
	}
	if err = indexTask.Await(context.Background()); err != nil {
		t.Fatalf("CreateIndex await failed: %v", err)
	}

	// 加载 collection
	loadTask, err := mc.Client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		t.Fatalf("LoadCollection failed: %v", err)
	}
	if err = loadTask.Await(ctx); err != nil {
		t.Fatalf("LoadCollection await failed: %v", err)
	}

	// 单向量搜索
	topK := 3
	singleVec := vectorsBatch[0]
	singleResults, err := mc.SearchSingle(collectionName, singleVec, topK)
	if err != nil {
		t.Fatalf("SearchSingle failed: %v", err)
	}
	t.Logf("Single search returned %d results", len(singleResults))

	// 批量搜索
	resultsSets := make([][]milvusclient.ResultSet, len(vectorsBatch))
	for i, vec := range vectorsBatch {
		res, err := mc.SearchSingle(collectionName, vec, topK)
		if err != nil {
			t.Fatalf("SearchSingle failed for text '%s': %v", texts[i], err)
		}
		resultsSets[i] = res
	}
	for i, res := range resultsSets {
		t.Logf("Text '%s' batch search returned %d results", texts[i], len(res))
	}

	// Retrieve / Aggregate
	retrieved, err := mc.RetrieveTextsByVector(collectionName, []entity.Vector{entity.FloatVector(singleVec)}, topK, "my_varchar")
	if err != nil {
		t.Fatalf("RetrieveTextsByVector failed: %v", err)
	}
	t.Logf("Retrieved texts: %+v", retrieved)

	aggregated, err := mc.AggregateTextsByVector(collectionName, []entity.Vector{entity.FloatVector(singleVec)}, topK, "my_varchar")
	if err != nil {
		t.Fatalf("AggregateTextsByVector failed: %v", err)
	}
	t.Logf("Aggregated text: %s", aggregated[0])
}