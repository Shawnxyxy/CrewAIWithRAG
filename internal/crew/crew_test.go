package crew

import (
	"testing"
	"time"

	"crew_ai_with_rag/internal/config"
	"crew_ai_with_rag/internal/llm"
	"crew_ai_with_rag/internal/tools/vector"
)

var llmCfg = &config.LLMConfig{
	APIType:   "openai",
	BaseURL:   "http://localhost:3000/v1",
	APIKey:    "sk-NSbOQNyTx8ZZFQsL60Fc927a74994fCeAe14724f2bAfAdAb",
	ModelName: "qwen-turbo",
	Timeout: 30*time.Second,
}

var embCfg = &config.EmbeddingConfig{
	APIType:   "openai",
	BaseURL:   "http://localhost:3000/v1",
	APIKey:    "sk-6kxLB9kLB7VW8jOxAe3aB261136e44989a31F19392162c4e",
	ModelName: "text-embedding-v1", // 阿里的embedding模型名
}

func TestCrewRetrievalToLlm(t *testing.T) {
	// ---- VectorStrore ----
	vs, err := vector.NewVectorStore("localhost:19530", embCfg, "crew_test_collection", "my_varchar", "my_vector", 3)
	if err != nil {
		t.Fatalf("NewVectorStore failed: %v", err)
	}
	if err := vs.EnsureCollection(1536); err != nil {
		t.Fatalf("EnsureCollection failed: %v", err)
	}
	texts := []string{"用户张三，男性，45岁，有高血压病史，长期服用降压药。",
	"用户李四，女性，32岁，近期出现失眠和焦虑症状。",
	"健康建议：高血压患者应注意低盐饮食，保持规律运动。",
	"心理健康建议：长期失眠可能与焦虑有关，建议进行心理咨询。",
	"今天深圳天气晴朗，最高气温28度。",
	"现在是12月24日的凌晨2点53分",
	"小鹏，男性，23岁，最近在和我交往暧昧"}
	if err := vs.InsertBatchTexts(texts); err != nil {
		t.Fatalf("InsertBatchTexts failed: %v", err)
	}
	// ---- Agents ----
	retrievalAgent := NewRetrievalAgent(vs)

	llmClient := llm.NewOpenAIClient(llmCfg)
	llmAgent := NewLLMAgent(llmCfg.ModelName, llmClient)
	// ---- Process ----
	process := NewSequentialProcess(
		retrievalAgent,
		llmAgent,
	)
	// ---- Task ----
	task := NewTask("rag_task", process)
	// ---- crew ----
	crew := NewCrew(task)
	// ---- 执行retrieval ----
	queries := []string{"高血压患者在日常生活中应该注意什么？",
	"长期失眠和焦虑可能带来哪些健康问题？",
	"深夜了，此时应该和小鹏说些什么比较好呢？"}
	results, err := crew.Run("rag_task", queries, 2)
	if err != nil {
		t.Fatalf("crew run failed: %v", err)
	}
	// ---- 打印结果 ----
	for i, r := range results {
		t.Logf("Query %d final output:\n%s", i, r)
	}
}