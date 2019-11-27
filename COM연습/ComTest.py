import win32com.client

excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = True
wb = excel.Workbooks.Open("C:\\Users\\taewoo\\Desktop\\input.xlsx")
ws = wb.ActiveSheet
ws.Cells(1, 2).Value = "is"
ws.Range("C1").Value = "good"
ws.Range("C1").Interior.ColorIndex = 10 # 칼라인덱스는 0~56값을 가진다.
ws.Range("A2:C2").Interior.ColorIndex = 27
