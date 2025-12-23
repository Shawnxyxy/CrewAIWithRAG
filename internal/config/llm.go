package config

import "time"

type LLMConfig struct {
	APIType   string `yaml:"api_type"`
	BaseURL   string `yaml:"base_url"`
	APIKey    string `yaml:"api_key"`
	ModelName string `yaml:"model_name"`
	Timeout   time.Duration    `yaml:"timeout"`
}