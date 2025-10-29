from src.config import Config
class DataManager:
    def __init__(self,datasets_dir,save_raw_dir,save_tok_dir):
        self.datasets_dir=datasets_dir
        self.save_raw_dir=save_raw_dir
        self.save_tok_dir=save_tok_dir
    def build_english2french_dataset(self):
        pass
def main():
    datasets_dir=Config.DATA_DIR
    save_raw_dir=Config.SAVE_RAW_DIR
    save_tok_dir=Config.SAVE_TOK_DIR
if __name__=='__main__':
    main()