package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"crew_ai_with_rag/internal/config"
)

type OpenAIClient struct {
	cfg *config.LLMConfig
}

type chatMessage struct {
	Role	string	`json:"role"`
	Content	string	`json:"content"`
}

type chatRequest struct {
	Model	string			`json:"model"`
	Message []chatMessage	`json:"messages"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

func NewOpenAIClient(cfg *config.LLMConfig) *OpenAIClient {
	return &OpenAIClient{cfg: cfg}
}

func (c *OpenAIClient) Generate(prompt string) (string, error) {
	reqBody := chatRequest{
		Model: c.cfg.ModelName,
		Message: []chatMessage{
			{
				Role:		"system",
				Content: 	"You are a helpful assistant.",
			},
			{
				Role:		"user",
				Content:	prompt,
			},
		},
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest(
		"POST",
		c.cfg.BaseURL+"/chat/completions",
		bytes.NewReader(data),
	)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)

	client := &http.Client{
		Timeout: c.cfg.Timeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("llm error: %s", string(body))
	}
	var respObj chatResponse
	if err := json.Unmarshal(body, &respObj); err != nil {
		return "", err
	}
	if len(respObj.Choices) == 0 {
		return "", fmt.Errorf("empty llm response")
	}
	return respObj.Choices[0].Message.Content, nil
}