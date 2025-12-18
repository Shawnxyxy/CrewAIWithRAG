package vector

import (
	"fmt"
	"context"
	"time"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"github.com/milvus-io/milvus/client/v2/column"
)

func (mc *MilvusClient) RetrieveTextsByVector(
	collectionName string,
	vectors []entity.Vector,
	topK int,
	textField string,
) ([][]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 1. 发起搜索
	resultsSets, err := mc.Client.Search(ctx, milvusclient.NewSearchOption(
		collectionName,
		topK,
		vectors,
	).WithANNSField("my_vector").
	  WithConsistencyLevel(entity.ClStrong).
	  WithOutputFields(textField))
	if err != nil {
		return nil, fmt.Errorf("milvus search failed: %v", err)
	}
	// 2.从结果中提取指定字段的文本
	results := make([][]string, len(resultsSets))
	for i, rs := range resultsSets {
		var textSet []string
		for _, col := range rs.Fields {
			if col.Name() != textField {
				continue
			}
			strCol, ok := col.(*column.ColumnVarChar) // 断言类型为 *ColumnVarChar
			if !ok {
				return nil, fmt.Errorf("filed '%s' is not varchar type", textField)
			}
			textSet = append(textSet, strCol.Data()...)
		}
		results[i] = textSet
	}
	return results, nil
}

func (mc *MilvusClient) AggregateTextsByVector(
	collectionName string,
	vectors []entity.Vector,
	topK int,
	textField string,
) ([]string, error) {
	resultsPerVector, err := mc.RetrieveTextsByVector(collectionName, vectors, topK, textField)
	if err != nil {
		return nil, fmt.Errorf("retrieve texts failed: %v", err)
	}
	aggregatedText := make([]string, len(resultsPerVector))
	for i, textSlice := range resultsPerVector {
		aggregatedText[i] = strings.Join(textSlice, "\n")
	}
	return aggregatedText, nil
}