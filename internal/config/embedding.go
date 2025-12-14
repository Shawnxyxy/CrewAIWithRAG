package config

import (
    "fmt"
    "os"
    "gopkg.in/yaml.v3"
)

type EmbeddingConfig struct {
    APIType   string `yaml:"api_type"`
    BaseURL   string `yaml:"base_url"`
    APIKey    string `yaml:"api_key"`
    ModelName string `yaml:"model_name"`
	Dimension int
}

func LoadEmbeddingConfig(path string) (*EmbeddingConfig, error) {
	data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("读取 YAML 文件失败: %v", err)
    }

    var cfg EmbeddingConfig
    err = yaml.Unmarshal(data, &cfg)
    if err != nil {
        return nil, fmt.Errorf("解析 YAML 文件失败: %v", err)
    }

    return &cfg, nil
}

