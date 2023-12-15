import uvicorn


if __name__ == "__main__":
    print(__name__.split('.')[0])
    uvicorn.run("webapp.backend.app.api:app", 
                host="127.0.0.1" , 
                port = 8000 , 
                reload=True)


