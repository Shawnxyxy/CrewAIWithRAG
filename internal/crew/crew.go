package crew

import (
	"fmt"

	"crew_ai_with_rag/internal/tools/vector"
)
// Agent统一接口
type Agent interface {
	Name() string
	Run(input []string, topK int) ([]string, error)
}

type Process interface {
	Run(input []string, topK int) ([]string, error)
}

type Crew struct {
	tasks map[string]*Task
}

type RetrievalAgent struct {
	VectorStore *vector.VectorStore
}

type LLMClient interface {
	Generate(prompt string) (string, error)
}

type LLMAgent struct {
	Model string
	Client LLMClient
}

type SequentialProcess struct {
	agents []Agent
}

type Task struct {
	Name 	string
	Process Process
}

func NewCrew(tasks ...*Task) *Crew {
	m := make(map[string]*Task)
	for _, t := range tasks {
		m[t.Name] = t
	}
	return &Crew{tasks: m}
}

func NewLLMAgent(model string, client LLMClient) *LLMAgent {
	return &LLMAgent{
		Model:  model,
		Client: client,
	}
}

func NewSequentialProcess(agents ...Agent) *SequentialProcess {
	return &SequentialProcess{agents: agents}
}

func NewTask(name string, process Process) *Task {
	return &Task{
		Name:	 name,
		Process: process,
	}
}

func (c *Crew) Run(taskName string, query []string, topK int) ([]string, error) {
	task, ok := c.tasks[taskName]
	if !ok {
		return nil, fmt.Errorf("task %s not found", taskName)
	}
	return task.Run(query, topK)
}

func NewRetrievalAgent(vs *vector.VectorStore) *RetrievalAgent {
	return &RetrievalAgent{VectorStore: vs}
}

func (a *RetrievalAgent) Name() string {
	return "retrieval_agent"
}

func (a *LLMAgent) Name() string {
	return "llm_agent"
}

func (a *RetrievalAgent) Run(query []string, topK int) ([]string, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("empty query")
	}
	return a.VectorStore.AggregateBatch(query, topK)
}

func (a *LLMAgent) Run(context []string, _ int) ([]string, error) {
	if len(context) == 0 {
		return nil, fmt.Errorf("empty context")
	}
	results := make([]string, len(context))
	for i, ctx := range context {
		resp, err := a.Client.Generate(ctx)
		if err != nil {
			return nil, err
		}
		results[i] = resp
	}
	return results, nil
}

func (p *SequentialProcess) Run(input []string, topK int) ([]string, error) {
	var err error
	output := input

	for _, agent := range p.agents {
		output, err = agent.Run(output, topK)
		if err != nil {
			return nil, fmt.Errorf("agent %s failed: %v", agent.Name(), err)
		}
	}
	return output, nil
}

func (t *Task) Run(input []string, topK int) ([]string, error) {
	return t.Process.Run(input, topK)
}