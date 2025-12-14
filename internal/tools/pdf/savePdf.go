package pdf

import (
	"fmt"
	"github.com/jung-kurt/gofpdf"
)

func SaveText2Pdf(text string, filename string) string {
	// 初始化pdf
	pdf := gofpdf.New("P", "mm", "A4", "")
	pdf.AddPage()
	// 添加和设置字体，使用支持中文的字体文件（如 simhei.ttf）
	pdf.AddUTF8Font("SimHei", "", "simhei.ttf")
	pdf.SetFont("SimHei", "", 16)
	// 添加标题
	pdf.CellFormat(0, 10, "", "", 1, "C", false, 0, "")
	// 写入文本内容
	pdf.SetFont("SimHei", "", 12)
	pdf.MultiCell(0, 10, text, "", "", false)
	// 保存pdf文件
	err := pdf.OutputFileAndClose("output/" + filename)
	if err != nil {
		return fmt.Sprintf("保存PDF文件时出错: %v", err)
	}
	return fmt.Sprintf("请前往 %s 查看报告", filename)
}