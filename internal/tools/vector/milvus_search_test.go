package vector

import (
	"context"
	"testing"
	"time"
	"fmt"

	"crew_ai_with_rag/internal/config"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// 测试用配置
	var embCfg = &config.EmbeddingConfig{
		APIType:   "openai",
		BaseURL:   "http://localhost:3000/v1",
		APIKey:    "sk-6kxLB9kLB7VW8jOxAe3aB261136e44989a31F19392162c4e",
		ModelName: "text-embedding-v1", // 阿里的embedding模型名
	}

func TestSearchInMilvus(t *testing.T) {
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

func TestBatchEmbedding(t *testing.T) {
	texts := []string {
		"健康档案检索测试",
		"用户健康状况分析",
		"生活方式与慢性疾病关系",
	}
	vectors, err := GenerateBatchEmbedding(embCfg, texts)
	if err != nil {
		t.Fatalf("GenerateBatchEmbedding failed: %v", err)
	}
	if len(vectors) != len(texts) {
		t.Fatalf("Expected %d vectors, got %d", len(texts), len(vectors))
	}
	for i, vec := range vectors {
		if len(vec) != embCfg.Dimension {
			t.Fatalf("Vector %d has unexpected dimension: %d", i, len(vec))
		}
		hasNonZero := false
		for _, v := range vec {
			if v != 0 {
				hasNonZero = true
				break
			} 
		}
		if !hasNonZero {
			t.Fatalf("Vector %d is all zeros", i)
		}
	}
}

// EnsureFreshCollection 检查指定 collection 是否存在，如果存在则删除，保证是干净的状态
func EnsureFreshCollection(ctx context.Context, mc *MilvusClient, collectionName string) error {
	exists, err := mc.Client.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		return fmt.Errorf("HasCollection check failed: %v", err)
	}
	if exists {
		if err := mc.Client.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName)); err != nil {
			return fmt.Errorf("DropCollection failed: %v", err)
		}
	}
	return nil
}

func TestSearchSingle(t *testing.T) {
	milvusAddr := "localhost:19530"
	collectionName := "health_records_test"
	// 1. 创建客户端
	mc, err := NewMilvusClient(milvusAddr)
	if err != nil {
		t.Fatalf("Failed to create Milvus client: %v", err)
	}
	// 2. 创建 Collection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = EnsureFreshCollection(ctx, mc, collectionName)
	if err != nil {
    	t.Fatalf("%v", err)
	}
	err = mc.CreateCollection(collectionName)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	// 3. 生成向量并插入collection
	queryText := "健康档案检索测试"
	vec, err := GetEmbedding(embCfg, queryText)
	if err != nil {
		t.Fatalf("GetEmbedding failed: %v", err)
	}
	insertOpt := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithInt64Column("my_id", []int64{1}).
		WithFloatVectorColumn("my_vector", len(vec), [][]float32{vec}).
		WithVarcharColumn("my_varchar", []string{"placeholder"})
	if _, err := mc.Client.Insert(context.Background(), insertOpt); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
	// 4. 创建索引
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
	// 5. 加载 Collection
	loadTask, err := mc.Client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		t.Fatalf("LoadCollection failed: %v", err)
	}
    if err = loadTask.Await(ctx); err != nil {
        t.Fatalf("collection load await failed: %v", err)
    }
	// 6.搜索
	topK := 3
	results, err := mc.SearchSingle("health_records_test", vec, topK)
	if err != nil {
		t.Fatalf("SearchSingle failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("No results returned from SearchSingle")
	}
	t.Logf("SearchSingle returned %d results", len(results))
}

func TestSearchBatch(t *testing.T) {
	milvusAddr := "localhost:19530"
	collectionName := "health_records_test"

	// 1. 创建客户端
	mc, err := NewMilvusClient(milvusAddr)
	if err != nil {
		t.Fatalf("Failed to create Milvus client: %v", err)
	}
	// 2. 创建 Collection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = EnsureFreshCollection(ctx, mc, collectionName)
	if err != nil {
    	t.Fatalf("%v", err)
	}
	err = mc.CreateCollection(collectionName)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}
	// 3. 批量生成向量并插入collection
    texts := []string{
        "健康档案检索测试",
        "用户健康状况分析",
        "今天的天气如何？",
    }
    vectorsBatch, err := GenerateBatchEmbedding(embCfg, texts)
    if err != nil {
        t.Fatalf("GenerateBatchEmbedding failed: %v", err)
    }
	ids := make([]int64, len(vectorsBatch))
	for i := range vectorsBatch {
		ids[i] = int64(i + 1)
	}
	varcharData := make([]string, len(vectorsBatch))
	for i := range varcharData {
    	varcharData[i] = "placeholder"
	}
	insertOpt := milvusclient.NewColumnBasedInsertOption(collectionName).
		WithInt64Column("my_id", ids).
		WithFloatVectorColumn("my_vector", len(vectorsBatch[0]), vectorsBatch).
		WithVarcharColumn("my_varchar", varcharData)
	if _, err := mc.Client.Insert(context.Background(), insertOpt); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
	// 4. 创建索引
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
	// 5. 加载 Collection
	loadTask, err := mc.Client.LoadCollection(context.Background(), milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		t.Fatalf("LoadCollection failed: %v", err)
	}
    if err = loadTask.Await(ctx); err != nil {
        t.Fatalf("collection load await failed: %v", err)
    }
    // 6.批量搜索
    topK := 3
    resultsSets := make([][]milvusclient.ResultSet, len(vectorsBatch))
    for i, vec := range vectorsBatch {
        res, err := mc.SearchSingle("health_records_test", vec, topK)
        if err != nil {
            t.Fatalf("SearchSingle failed for text '%s': %v", texts[i], err)
        }
        resultsSets[i] = res
    }
    // 检查结果
    for i, res := range resultsSets {
        if len(res) == 0 {
            t.Fatalf("Expected non-empty search results for text '%s'", texts[i])
        }
        t.Logf("Text '%s' search returned %d results", texts[i], len(res))
    }
}