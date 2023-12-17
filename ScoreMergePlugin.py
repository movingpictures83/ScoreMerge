#pickle PickleJar/softmax.pkl
#csvfile data/capri_score/CAPRI_scores_targets.csv
#pid     model
#splittwice      true
#on      pid
#how     left
#dMaSIF  avg_score
#pid
#DeepRank_GNN
#GNN_DOVE

import PyPluMA
import PyIO
import pandas as pd
import pickle
class ScoreMergePlugin:
    def input(self, inputfile):
       infile = open(inputfile, 'r')
       self.parameters = dict()
       self.mapping = dict()
       self.models = []
       self.parameters["pickle"] = ""
       self.parameters["csvfile"] = ""
       self.parameters["pid"] = ""
       self.parameters["PPI"] = ""
       self.parameters["split"] = ""
       self.parameters["splittwice"] = ""
       self.parameters["on"] = ""
       self.parameters["how"] = ""
       for line in infile:
           line = line.strip()
           if ('\t' in line):
              contents = line.strip().split('\t')
              key = contents[0]
              if (key in self.parameters):
                  self.parameters[key] = contents[1]
              else:
                  self.mapping[key] = contents[1]
           else:
               self.models.append(line)

       
    def run(self):
        pass

    def output(self, outputfile):
        inpickle = open(PyPluMA.prefix()+"/"+self.parameters["pickle"], "rb")
        df = pickle.load(inpickle)
        csvfile = PyPluMA.prefix()+"/"+self.parameters["csvfile"]
        if (csvfile.endswith('csv') or csvfile.endswith('txt')):
         df_d = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["csvfile"])
         if ('pid' in self.parameters and self.parameters["pid"] != ""):
          if (self.parameters["splittwice"] == "true"):
           df_d['pid'] = df_d[self.parameters["pid"]].apply(lambda x: x.split('_')[0] + '-' + x.split('_')[1])
          else:
           df_d['pid'] = df_d[self.parameters["pid"]].apply(lambda x: x.split('_')[0])
         else: # PPI
          if ("split" in self.parameters and self.parameters["split"] == "true"):
           df_d['PPI'] = df_d[self.parameters["PPI"]].apply(lambda x: x+'_A_Z')
         for key in self.mapping:
            df_d[key] = df_d[self.mapping[key]]
         df_d = df_d[self.models]
         if ("how" in self.parameters and self.parameters["how"] != ""):
           df = df.merge(df_d, on=self.parameters["on"], how=self.parameters["how"])
         else:
           df = df.merge(df_d, on=self.parameters["on"])
        else: #pickle
         inpickle2 = open(PyPluMA.prefix()+"/"+self.parameters["csvfile"], "rb")
         output2 = pickle.load(inpickle2)
         pred_probabilities = list(output2.cpu().detach().numpy())
         df[self.models[0]] = pred_probabilities
        outfile = open(outputfile, "wb")
        pickle.dump(df, outfile)
