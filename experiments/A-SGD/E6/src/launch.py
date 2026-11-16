import pathlib,sys
R=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'shared/src'))
import round3
if __name__=='__main__':round3.main('E6')
