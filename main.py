#Fecha creación 03/06/2021 a las 11:53
#Por Esteban Cruz López

#Variable fundamental del método de clasificación
K=10

#Margen de similitud
margenSimilitud = 0.01

#Variable global para manejar la colección de entrenamiento
trainingSet=[]
testSet=[]

#Variable donde se almacenará los resultados del algoritmo de clasificación
nuevasClases=[]

#Estructura de dato de cada entrada de la colección
class Documento:
    def __init__(self,docId,clase,numTerminos,pesosXTermino):
        self.docId=docId
        self.clase=clase
        self.numTerminos=numTerminos
        self.pesosXTermino=dict([(x.split("/")[0],float(x.split("/")[1])) for x in pesosXTermino.split(" ")])
        

#Se llena la lista de la colección con los documentos extraídos de training-set.csv
with open ("training-set.csv","r")as trainingDoc:
    trainingDoc.readline()
    trainingSet=[Documento(*linea.split("\t")) for linea in trainingDoc.readlines()]

#Se abre el archivo de la colección de prueba y por cada entrada se ejecuta el algoritmo de clasificación    
with open ("test-set.csv","r") as testDoc:
    testDoc.readline()
    testSet=[Documento(*linea.split("\t")) for linea in testDoc.readlines()]

def realizarConsulta(testDoc):
    similitudes={}
    for trainingDoc in trainingSet:
        similitudes[trainingDoc]=0
        for termino,peso in testDoc.pesosXTermino.items():
            if termino in trainingDoc.pesosXTermino:
                similitudes[trainingDoc]+=peso*trainingDoc.pesosXTermino[termino]
    #Se crea el escalafón y se ordena de manera descendente
    escalafon=list(similitudes.items())
    escalafon.sort(key=lambda x:x[1],reverse=True)
    return escalafon 
    
print("DocId","Clase KNN",sep="\t")
for testDoc in testSet:
    #Se obtiene el escalafón
    escalafon=realizarConsulta(testDoc)
    
    clases={}
    #Se analizan los K más relevantes, se extrae la similitud por clase
    for doc,sim in escalafon[:K]:
        clases[doc.clase] = [sim] + (clases[doc.clase] if doc.clase in clases else [])
    #Se hace el promedio de similitudes por clase
    promedioPorClase=[(sum(x)/len(x),y) for y,x in clases.items()]
    promedioPorClase.sort(reverse=True)
    #Y se escoje el de promedio más alto
    clasesPosibles = [promedioPorClase[0]]
    for clasePosible in promedioPorClase[1:]:
        if clasesPosibles[0][0] - clasePosible[0] < margenSimilitud:
            clasesPosibles.append(clasePosible)
    if len(clasesPosibles) > 1:
        while(True):
            print(f"El documento con el id {testDoc.docId} necesita que le seleccione una categoria")
            contador = 1
            for clasePosible in clasesPosibles:
                print(f"{contador}. {clasePosible[1]}")
                contador += 1
            seleccion = input()
            if seleccion.isnumeric():
                seleccionInt = int(seleccion) - 1
                if seleccionInt < len(clasesPosibles):
                    testDoc.clase = clasesPosibles[int(seleccion) - 1][1]
                    break
    else:
        testDoc.clase = clasesPosibles[0][1]
    #nuevasClases+=[promedioPorClase[0][1]]
    print(testDoc.docId,testDoc.clase,sep="\t") 
    trainingSet.append(testDoc)
#Hacer el análisis de resultados