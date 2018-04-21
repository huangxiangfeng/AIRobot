# 曹雪龙 18600961258 微信同号
# 改变标准输出的默认编码
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding = 'utf8')

import tkinter as tk
from tkinter import *

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        # 用户名
        self.helloLabel = tk.Label(self, text='用户名')
        self.helloLabel.pack()
        self.nameInput = tk.Entry(self)
        self.nameInput.pack()
        # 用户密码
        self.helloLabel = tk.Label(self, text='密码')
        self.helloLabel.pack()
        self.passwordInput = tk.Entry(self)
        self.passwordInput.pack()
        # 登录按钮
        self.alertButton = tk.Button(self, text='登录', command=self.hello)
        self.alertButton.pack()
        # 登录后，将用户信息存储下来，再次登录进行对比
        # 退出按钮
        self.quit = tk.Button(self, text="取消", fg="red",
                              command= root.destroy)
        self.quit.pack(side="bottom")
        # 人脸识别
        self.faceButton = tk.Checkbutton(self, text='人脸识别')
        self.faceButton.pack()

        # 语音识别-语音聊天
        self.voiceButton = tk.Checkbutton(self, text='语音聊天')
        self.voiceButton.pack()
		
        self.dialogueContent = tk.Text(self, width=50,height=20)
        self.dialogueContent.pack()
        self.dialogueContent.insert(INSERT,"bot: ",'big')
        self.dialogueContent.insert(INSERT,"first !\n")
        self.dialogueContent.insert(INSERT,"second ! :me")
		

        # 发言文本框
        self.dialogueInput = tk.Entry(self)
        self.dialogueInput.pack()
        # 发言文本发送按钮
        self.dialogueSendButton = tk.Button(self, text='发送', command=self.dialogueSend)
        self.dialogueSendButton.pack()

    def hello(self):
        self.master.title('聊天机器人1111111111')
        name = self.nameInput.get() or 'world'
        if (self.faceButton.getboolean ):
            name = name + ',请进行人脸识别'
        if (self.voiceButton.getboolean ):
            name = name + ',可以和机器人进行语音聊天！'
        else:
            name = name + ',只能和机器人进行文本聊天！'
        messagebox.showinfo('登录提示', 'Hello, %s' % name)

    def dialogueSend(self):
        self.dialogueContent.insert(INSERT,"me:"+self.dialogueInput.get()+"\n")
		
#app = Application()
root = tk.Tk()
root.geometry('500x600')#窗体大小
root.resizable(False, False)#固定窗体
app = Application(master=root)
# 设置窗口标题:
app.master.title('聊天机器人')
# 主消息循环:
app.mainloop()

# 待解决的问题
# 1、窗口要固定大小
# 2、窗口内控件的布局