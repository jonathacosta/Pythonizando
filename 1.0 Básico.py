# -*- coding: utf-8 -*-
"""
Código básico de utilização do Tkinter
Créditos Rafael Serafim -  https://www.youtube.com/watch?v=PYHVFppB8PI&list=PLqx8fDb-FZDFznZcXb_u_NyiQ7Nai674-&index=8
"""
from tkinter import *
from tkinter import ttk


class Funcs():
    '''
    Funções de Back-end
    '''
    def limpa_tela(self):
        self.codigo_entry.delete(0,END)
        self.nome_entry.delete(0,END)
        self.fone_entry.delete(0,END)
        self.cidade_entry.delete(0,END)

class Application(Funcs):
    '''
    Funções de front-end
    '''
    def __init__(self,root):
        '''
        Método inicia o loop para interface do usuário e chama métodos
        '''
        self.root = root
        self.tela()
        self.frames_de_tela()
        self.widgets_frame1()
        self.lista_frame2()
        root.mainloop() 
        
    def tela(self):
        '''
        Método define título, cor e tamanhos da tela.
        '''
        self.root.title('Cargas')
        self.root.configure(background='gray')
        self.root.geometry("700x500")
        self.root.resizable(True,True)
        self.root.maxsize(width= 900,height= 700)
        self.root.minsize(width= 500,height= 300)
    
    def frames_de_tela(self):
        '''
        Áreas de plano enquadradas no objeto passado 'root'
        Tipos comuns de frame
        1. pack - o mais comum, mas limitado em posições e funções
        2. grid - utiliza o conceito e planhila
        3. place - utilizado com coordenado absoluta ou relativa
        '''
        self.frame_1 = Frame(self.root, bd = 4, bg = '#dfe3ee',highlightbackground = "#759fe6", highlightthickness=6)
        self.frame_1.place(relx= 0.02, rely= 0.02, relwidth = 0.96, relheight= 0.46) # 10% esquerda e 10% abaixo
        
        self.frame_2 = Frame(self.root, bd = 4, bg = '#dfe3ee',highlightbackground = "#759fe6", highlightthickness=6)
        self.frame_2.place(relx= 0.02, rely= 0.5, relwidth = 0.96, relheight= 0.46) # 10% esquerda e 10% abaixo
       
        
    def widgets_frame1(self):
        '''
        Método cria os botões para o frame1, os labels para os botões e as entradas
        '''
        
        self.bt_limpar = Button(self.frame_1, text = 'Limpar', 
                                bd=2,bg='#107db2',fg='white',font=('verdana',10,'bold'),
                                command=self.limpa_tela)
        self.bt_limpar.place(relx = 0.2, rely = 0.1,relwidth = 0.1, relheight = 0.15)
        
        self.bt_buscar = Button(self.frame_1, text = 'Buscar', bd=2,bg='#107db2',fg='white',font=('verdana',10,'bold'))
        self.bt_buscar.place(relx = 0.3, rely = 0.1,relwidth = 0.1, relheight = 0.15)
        
        self.bt_novo = Button(self.frame_1, text = 'Novo', bd=2,bg='#107db2',fg='white',font=('verdana',10,'bold'))
        self.bt_novo.place(relx = 0.6, rely = 0.1,relwidth = 0.1, relheight = 0.15)
        
        self.bt_alterar = Button(self.frame_1, text = 'Alterar', bd=2,bg='#107db2',fg='white',font=('verdana',10,'bold'))
        self.bt_alterar.place(relx = 0.7, rely = 0.1,relwidth = 0.1, relheight = 0.15)
        
        
        self.bt_apagar  = Button(self.frame_1, text = 'Apagar', bd=2,bg='#107db2',fg='white',font=('verdana',10,'bold'))
        self.bt_apagar.place(relx = 0.8, rely = 0.1,relwidth = 0.1, relheight = 0.15)
        
        # Criação de label e entrada do código
        self.lb_codigo = Label(self.frame_1, text = 'Código',font=('verdana',10,'bold') )
        self.lb_codigo.place(relx= 0.05, rely= 0.05)     
        self.codigo_entry = Entry(self.frame_1)
        self.codigo_entry.place(relx = 0.05, rely = 0.15, relwidth=0.08)
        
        # Criação de label e entrada do nome
        self.lb_nome = Label(self.frame_1, text = 'Nome',font=('verdana',10,'bold'))
        self.lb_nome.place(relx= 0.05, rely= 0.35)        
        self.nome_entry = Entry(self.frame_1)
        self.nome_entry.place(relx = 0.05, rely = 0.45, relwidth=0.8)
        
        # Criação de label e entrada do telefone
        self.lb_fone = Label(self.frame_1, text = 'Telefone',font=('verdana',10,'bold'))
        self.lb_fone.place(relx= 0.05, rely= 0.55)
        self.fone_entry = Entry(self.frame_1)
        self.fone_entry.place(relx = 0.05, rely = 0.65, relwidth=0.3)
        
        # Criação de label e entrada do cidade
        self.lb_cidade = Label(self.frame_1, text = 'Cidade',font=('verdana',10,'bold'))
        self.lb_cidade.place(relx= 0.45, rely= 0.55)
        self.cidade_entry = Entry(self.frame_1)
        self.cidade_entry.place(relx = 0.45, rely = 0.65, relwidth=0.3)
        
    def lista_frame2(self):
        self.listaCli = ttk.Treeview(self.frame_2, height= 3, column=('col','col2','col3','col4'))
        self.listaCli.heading('#0',text="")
        self.listaCli.heading('#1',text="Código")
        self.listaCli.heading('#2',text="Nome")
        self.listaCli.heading('#3',text="Telefone")
        self.listaCli.heading('#4',text="Cidade")
        
        self.listaCli.column('#0',width=1)
        self.listaCli.column('#1',width=50)
        self.listaCli.column('#2',width=200)
        self.listaCli.column('#3',width=125)
        self.listaCli.column('#4',width=125)
        
        self.listaCli.place(relx = 0.01, rely = 0.01, relwidth = 0.95, relheight = 0.85)
        
        self.scrolLista = Scrollbar(self.frame_2,orient='vertical')
        self.listaCli.configure(yscroll=self.scrolLista.set)
        self.scrolLista.place(relx=0.96, rely=0.1, relwidth=0.04, relheight=0.9)
        

           
ihm = Tk()  
Application(root=ihm)        