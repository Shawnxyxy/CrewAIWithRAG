package pdf

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const maxChunkLen = 170 // 170个中文 × 3bytes ≈ 510bytes(Milvus 的 my_varchar 字段，最大长度是 512)

func ReadTxtToTexts(txtPath string) ([]string, error) {
	file, err := os.Open(txtPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open TXT file: %v", err)
	}

	defer file.Close()

	var results []string
	var buffer []string

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// 空行作为一个段落的分割
		if line == "" {
			if len(buffer) > 0 {
				para := strings.Join(buffer, " ")
        		chunks := splitToChunks(para, maxChunkLen)
        		results = append(results, chunks...)
        		buffer = nil
			}
			continue
		}
		buffer = append(buffer, line)
	}
	// 处理文件末尾
	if len(buffer) > 0 {
		para := strings.Join(buffer, " ")
		chunks := splitToChunks(para, maxChunkLen)
		results = append(results, chunks...)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read TXT file: %v", err)
	}

	return results, nil
}

func ReadDirTxts(dir string) ([]string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("getwd failed: %v", err)
	}
	projectRoot, err := filepath.Abs(filepath.Join(cwd, "..", ".."))
	if err != nil {
		return nil, fmt.Errorf("get project root failed: %v", err)
	}

	absDir := filepath.Join(projectRoot, dir)
	fmt.Println("Reading TXT files from directory:", absDir)

	var allTexts []string

	err = filepath.Walk(absDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if (filepath.Ext(path) == ".txt") {
			texts, err := ReadTxtToTexts(path)
			if err != nil {
				return err
			}
			allTexts = append(allTexts, texts...)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(allTexts) == 0 {
		return nil, fmt.Errorf("no txt files found in %s", absDir)
	}
	return allTexts, nil
}

func splitToChunks(text string, maxLen int) []string {
	var chunks []string
	runes := []rune(text)

	for start := 0; start < len(runes); start += maxLen {
		end := start + maxLen
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[start:end]))
	}
	return chunks
}