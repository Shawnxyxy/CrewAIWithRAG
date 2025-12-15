package vector

import (
	"testing"
	"log"
)

const MilvusAddr = "localhost:19530"

// 验证 Milvus 客户端能否成功连接 Milvus 服务端
func TestNewMilvusClient(t *testing.T) {
	client, err := NewMilvusClient(MilvusAddr)
	if err != nil {
		t.Fatalf("Failed to connect Milvus client: %v", err)
	}
	log.Printf("Milvus client connected: %+v", client)
}

// 验证 创建 Collection 是否正常
func TestCreateCollection(t *testing.T) {
	// 创建客户端
	client, err := NewMilvusClient(MilvusAddr)
	if err != nil {
		t.Fatalf("failed to create Milvus client: %v", err)
	}

	// 测试 Collection 名称
	collectionName := "health_records_test"

	// 调用创建 Collection
	err = client.CreateCollection(collectionName)
	if err != nil {
		t.Fatalf("CreateCollection failed: %v", err)
	}

	t.Logf("Collection %s created successfully", collectionName)
}