package vector

import (
	"context"
	"fmt"
	"strings"
	"time"

	"crew_ai_with_rag/internal/config"

	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// VectorStore 是对外统一的工具类，封装 Milvus 操作
type VectorStore struct {
	Client      *MilvusClient
	EmbCfg      *config.EmbeddingConfig
	Collection  string
	TextField   string
	VectorField string
	TopKDefault int
}

// NewVectorStore 创建一个 VectorStore 实例
func NewVectorStore(addr string, embCfg *config.EmbeddingConfig,
	collection, textField, vectorField string, topK int) (*VectorStore, error) {
	mc, err := NewMilvusClient(addr)
	if err != nil {
		return nil, fmt.Errorf("failed to create to Milvus client: %v", err)
	}
	return &VectorStore{
		Client:      mc,
		EmbCfg:      embCfg,
		Collection:  collection,
		TextField:   textField,
		VectorField: vectorField,
		TopKDefault: topK,
	}, nil
}

// EnsureCollection 清理已有 collection 并创建新的
func (vs *VectorStore) EnsureCollection(dim int) error {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	// 检查是否存在，存在就删除
	exists, err := vs.Client.Client.HasCollection(ctx, milvusclient.NewHasCollectionOption(vs.Collection))
	if err != nil {
		return fmt.Errorf("HasCollection check failed: %v", err)
	}
	if exists {
		if err := vs.Client.Client.DropCollection(ctx, milvusclient.NewDropCollectionOption(vs.Collection)); err != nil {
			return fmt.Errorf("DropCollection failed: %v", err)
		}
	}

	// 创建 collection
	if err := vs.Client.CreateCollection(vs.Collection, dim); err != nil {
		return fmt.Errorf("CreateCollection failed: %v", err)
	}

	// 创建索引
	indexTask, err := vs.Client.Client.CreateIndex(context.Background(),
		milvusclient.NewCreateIndexOption(vs.Collection, vs.VectorField, index.NewAutoIndex(entity.COSINE)))
	if err != nil {
		return fmt.Errorf("CreateIndex failed: %v", err)
	}
	if err = indexTask.Await(context.Background()); err != nil {
		return fmt.Errorf("CreateIndex await failed: %v", err)
	}

	// 加载 collection
	loadTask, err := vs.Client.Client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(vs.Collection))
	if err != nil {
		return fmt.Errorf("LoadCollection failed: %v", err)
	}
	if err = loadTask.Await(ctx); err != nil {
		return fmt.Errorf("collection load await failed: %v", err)
	}

	return nil
}

// InsertText 插入单条文本
func (vs *VectorStore) InsertText(id int64, text string) error {
	vec, err := GetEmbedding(vs.EmbCfg, text)
	if err != nil {
		return fmt.Errorf("failed to get embedding: %v", err)
	}

	insertOpt := milvusclient.NewColumnBasedInsertOption(vs.Collection).
		WithInt64Column("my_id", []int64{id}).
		WithFloatVectorColumn(vs.VectorField, len(vec), [][]float32{vec}).
		WithVarcharColumn(vs.TextField, []string{text})

	if _, err := vs.Client.Client.Insert(context.Background(), insertOpt); err != nil {
		return fmt.Errorf("insert failed: %v", err)
	}

	return nil
}

// InsertBatchTexts 插入多条文本
func (vs *VectorStore) InsertBatchTexts(texts []string) error {
	vectors, err := GenerateBatchEmbedding(vs.EmbCfg, texts)
	if err != nil {
		return fmt.Errorf("batch embedding failed: %v", err)
	}

	ids := make([]int64, len(vectors))
	for i := range vectors {
		ids[i] = int64(i + 1)
	}

	insertOpt := milvusclient.NewColumnBasedInsertOption(vs.Collection).
		WithInt64Column("my_id", ids).
		WithFloatVectorColumn(vs.VectorField, len(vectors[0]), vectors).
		WithVarcharColumn(vs.TextField, texts)

	if _, err := vs.Client.Client.Insert(context.Background(), insertOpt); err != nil {
		return fmt.Errorf("insert batch failed: %v", err)
	}

	return nil
}

// SearchSingle 根据单条向量搜索
func (vs *VectorStore) SearchSingle(query string, topK int) ([][]string, error) {
	vec, err := GetEmbedding(vs.EmbCfg, query)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %v", err)
	}

	return vs.Client.RetrieveTextsByVector(vs.Collection, []entity.Vector{entity.FloatVector(vec)}, topK, vs.TextField)
}

// SearchBatch 根据多条向量搜索
func (vs *VectorStore) SearchBatch(queries []string, topK int) ([][]string, error) {
	vectors, err := GenerateBatchEmbedding(vs.EmbCfg, queries)
	if err != nil {
		return nil, fmt.Errorf("batch embedding failed: %v", err)
	}
	vecs := make([]entity.Vector, len(vectors))
	for i, v := range vectors {
		vecs[i] = entity.FloatVector(v)
	}

	return vs.Client.RetrieveTextsByVector(vs.Collection, vecs, topK, vs.TextField)
}

// AggregateSingle 搜索并聚合文本为字符串
func (vs *VectorStore) AggregateSingle(query string, topK int) (string, error) {
	results, err := vs.SearchSingle(query, topK)
	if err != nil {
		return "", err
	}
	return strings.Join(results[0], "\n"), nil
}

// AggregateBatch 搜索并聚合多条文本
func (vs *VectorStore) AggregateBatch(queries []string, topK int) ([]string, error) {
	results, err := vs.SearchBatch(queries, topK)
	if err != nil {
		return nil, err
	}
	aggregated := make([]string, len(results))
	for i, texts := range results {
		aggregated[i] = strings.Join(texts, "\n")
	}
	return aggregated, nil
}