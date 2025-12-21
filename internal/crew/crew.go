package crew

import (
	"fmt"

	"crew_ai_with_rag/internal/config"
	"crew_ai_with_rag/internal/tools/vector"
)

type Crew struct {
	VectorStore *vector.VectorStore
	Embedding *config.EmbeddingConfig
}

func NewCrew(vs *vector.VectorStore, embCfg *config.EmbeddingConfig) *Crew {
	return &Crew {
		VectorStore: vs,
		Embedding:   embCfg,
	}
}

// BuildContext
// 输入：用户 query
// 输出：可直接喂给 LLM 的上下文文本
func (c *Crew) BuildContext(query []string, topK int) ([]string, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("query is empty")
	}
	contexts, err := c.VectorStore.AggregateBatch(query, topK)
	if err != nil {
		return nil, fmt.Errorf("AggregateBatch fail:%v", err)
	}
	return contexts, nil
}