package main

import (
	"fmt"
	"log"

	"crew_ai_with_rag/internal/config"
	"crew_ai_with_rag/internal/crew"
	"crew_ai_with_rag/internal/llm"
	"crew_ai_with_rag/internal/tools/pdf"
	"crew_ai_with_rag/internal/tools/vector"
)

func main() {
	fmt.Println("Multi-Agent RAG System Starting...")

	// ---- Config ----
	llmCfg := &config.LLMConfig{
		APIType:   "openai",
		BaseURL:   "http://localhost:3000/v1",
		APIKey:    "sk-NSbOQNyTx8ZZFQsL60Fc927a74994fCeAe14724f2bAfAdAb",
		ModelName: "qwen-turbo",
	}

	embCfg := &config.EmbeddingConfig{
		APIType:   "openai",
		BaseURL:   "http://localhost:3000/v1",
		APIKey:    "sk-6kxLB9kLB7VW8jOxAe3aB261136e44989a31F19392162c4e",
		ModelName: "text-embedding-v1",
	}

	// ---- Vector Store (RAG) ----
	vs, err := vector.NewVectorStore(
		"localhost:19530",
		embCfg,
		"crew_prod_collection",
		"my_varchar",
		"my_vector",
		3,
	)
	if err != nil {
		log.Fatal(err)
	}

	if err := vs.EnsureCollection(1536); err != nil {
		log.Fatal(err)
	}

	texts, err := pdf.ReadDirTxts("internal/crew/input")
	if err != nil {
		log.Fatal(err)
	}

	if err := vs.InsertBatchTexts(texts); err != nil {
		log.Fatal(err)
	}

	// ---- Agents ----
	retrievalAgent := crew.NewRetrievalAgent(vs)
	llmClient := llm.NewOpenAIClient(llmCfg)
	llmAgent := crew.NewLLMAgent(llmCfg.ModelName, llmClient)
	reportAgent := crew.NewReportAgent(llmClient)

	// ---- Process / Task / Crew ----
	process := crew.NewSequentialProcess(
		retrievalAgent,
		llmAgent,
		reportAgent,
	)

	task := crew.NewTask("report_task", process)
	c := crew.NewCrew(task)

	// ---- Run ----
	queries := []string{
		"高血压患者在日常生活中应该注意什么？",
	}

	results, err := c.Run("report_task", queries, 1)
	if err != nil {
		log.Fatal(err)
	}

	for _, r := range results {
		fmt.Println("Report generated:")
		fmt.Println(r)
	}

	fmt.Println("System Finished")
}