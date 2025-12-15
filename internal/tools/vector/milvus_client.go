package vector

import (
	"fmt"
	"context"
	"time"
	
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/client/v2/entity"
)

type MilvusClient struct {
	Client *milvusclient.Client
}

func NewMilvusClient(addr string) (*MilvusClient, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
 	defer cancel()

 	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
 		Address: addr,
 	})
 	if err != nil {
 		return nil, fmt.Errorf("failed to connect Milvus client: %v", err)
 	}
	return &MilvusClient{Client: cli}, nil
}

func (mc *MilvusClient) CreateCollection(collectionName string) error {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
	// 创建 Collection 的 Schema
    schema := entity.NewSchema().
        WithDynamicFieldEnabled(true).
        WithField(entity.NewField().
            WithName("my_id").
            WithIsAutoID(false).
            WithDataType(entity.FieldTypeInt64).
            WithIsPrimaryKey(true)).
        WithField(entity.NewField().
            WithName("my_vector").
            WithDataType(entity.FieldTypeFloatVector).
            WithDim(5)).
        WithField(entity.NewField().
            WithName("my_varchar").
            WithDataType(entity.FieldTypeVarChar).
            WithMaxLength(512))

    // indexOptions 可以在这里定义
    var indexOptions []milvusclient.CreateIndexOption

	// 创建 Collection，并设置分片数为 1，启用 mmap
    collOption := milvusclient.NewCreateCollectionOption(collectionName, schema).
        WithIndexOptions(indexOptions...).
        WithShardNum(1).                                      // 设置分片数
        WithProperty("mmap_enabled", true)                    // 启用 mmap

	// 创建 Collection
    err := mc.Client.CreateCollection(ctx, collOption)
    if err != nil {
        return fmt.Errorf("create collection failed: %v", err)
    }

	// 获取 Collection 加载状态
    state, err := mc.Client.GetLoadState(ctx, milvusclient.NewGetLoadStateOption(collectionName))
    if err != nil {
        return fmt.Errorf("get load state failed: %v", err)
    }

    fmt.Printf("Collection load state: %+v\n", state.State)
    return nil
}
