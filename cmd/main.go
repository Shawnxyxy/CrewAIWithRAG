package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"crew_ai_with_rag/internal/config"
	"crew_ai_with_rag/internal/crew"
	"crew_ai_with_rag/internal/llm"
	"crew_ai_with_rag/internal/tools/pdf"
	"crew_ai_with_rag/internal/tools/vector"
)

func main() {
	fmt.Println("Multi-Agent RAG System Starting...")
	// ---- CLI flags ----
	queryFlag := flag.String("query", "", "Input query for the Agent system")
	topK := flag.Int("topk", 3, "Top-K retrieval results")
	flag.Parse()

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
		*topK,
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

	// ---- Run mode ----
	if *queryFlag != "" {
		runOnce(c, *queryFlag)
	} else {
		runInteractive(c)
	}

	fmt.Println("System Finished")
}

func runOnce(c *crew.Crew, query string) {
	results, err := c.Run("report_task", []string{query}, 1)
	if err != nil {
		log.Fatal(err)
	}

	for _, r := range results {
		fmt.Println("\n===== Report Generated =====")
		fmt.Println(r)
	}
}

func runInteractive(c *crew.Crew) {
	reader := bufio.NewReader(os.Stdin) // 读取当前终端输入流
	fmt.Println("Enter your query (type 'exit' to quit):")

	for {
		fmt.Print("> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Println("read input failed:", err)
			continue
		}

		query := strings.TrimSpace(input)
		if query == "" {
			continue
		}
		if strings.EqualFold(query, "exit") {
			break
		}

		runOnce(c, query)
	}
}