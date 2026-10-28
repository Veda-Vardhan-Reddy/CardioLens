from ml_engine.pipeline import train
if __name__=="__main__":
 print("Training comparative models...")
 print(train().to_string(index=False,float_format=lambda x:f"{x:.4f}"))
 print("Saved model, metadata and reports.")
