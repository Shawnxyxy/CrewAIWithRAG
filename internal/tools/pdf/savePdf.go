package pdf

import (
    "fmt"
    "os"
    "path/filepath"

    "github.com/jung-kurt/gofpdf/v2"
)

func SaveText2Pdf(text, filename string) string {
	cwd, _ := os.Getwd()
	fmt.Println("当前工作目录:", cwd)
	projectRoot, err := filepath.Abs(filepath.Join(cwd, "..", ".."))
	if err != nil {
    	return fmt.Sprintf("获取项目根目录失败: %v", err)
	}
	fontPath, err := filepath.Abs(
	filepath.Join(projectRoot, "internal", "tools", "pdf", "fonts", "SourceHanSansSC-VF.ttf"))
	fmt.Println("字体绝对路径:", fontPath)
	if err != nil {
		return fmt.Sprintf("获取字体绝对路径失败: %v", err)
	}
	// 再检查一次
	if _, err := os.Stat(fontPath); err != nil {
		return fmt.Sprintf("字体文件不存在: %v", err)
	}

    // 输出目录
    outputDir := filepath.Join("internal", "crew", "output") // 相对路径
    if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
        return fmt.Sprintf("创建 output 目录失败: %v", err)
    }
    outputPath := filepath.Join(outputDir, filename)
	fmt.Println("PDF 绝对路径:", outputPath)
	// 	读字体到内存
	fontBytes, err := os.ReadFile(fontPath)
	if err != nil {
		return fmt.Sprintf("读取字体文件失败: %v", err)
	}
    pdf := gofpdf.New("P", "mm", "A4", "")
    pdf.AddPage()
	// 用字节注册
    fontName := "sourcehansanssc-vf" // 用小写
    pdf.AddUTF8FontFromBytes(fontName, "", fontBytes)
    pdf.SetFont(fontName, "", 16)

    pdf.CellFormat(0, 10, "报告", "", 1, "C", false, 0, "")
    pdf.SetFont(fontName, "", 12)
    pdf.MultiCell(0, 10, text, "", "", false)

    if err := pdf.OutputFileAndClose(outputPath); err != nil {
        return fmt.Sprintf("保存 PDF 时出错: %v", err)
    }

    return fmt.Sprintf("请前往 %s 查看报告", outputPath)
}