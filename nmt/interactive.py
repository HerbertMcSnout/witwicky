import os
from logging import NullHandler
import torch
import subprocess
from io import StringIO
import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations

class InteractiveTranslator(object):
    def __init__(self, args):
        super(InteractiveTranslator, self).__init__()
        print("Initializing...")
        self.config = configurations.get_config(args.proto, getattr(configurations, args.proto), args.config_overrides)
        self.logger = ut.get_logger(logfile=None)
        
        self.model_file = args.model_file
        if self.model_file is None:
            self.model_file = os.path.join(self.config['save_to'], '{}.pth'.format(self.config['model_name']))

        if not os.path.exists(self.model_file):
            raise ValueError('Model file does not exist')

        self.data_manager = DataManager(self.config)

        self.src_formatter = self.maybe_spawn_process(args.src_formatter)
        self.trg_formatter = self.maybe_spawn_process(args.trg_formatter)

        self.translate()

    def maybe_spawn_process(self, cmd):
      if cmd:
        return subprocess.Popen(cmd,
                                bufsize=1,
                                shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

    def maybe_terminate_process(self, process):
      if process:
        process.terminate()

    def format_h(self, txt, process):
      if process:
        out, err = process.communicate(txt.encode())
        out = out.decode('utf-8').strip('\n')
        return out
      else:
        return txt

    def format_src(self, txt):
      return self.format_h(txt, self.src_formatter)
    
    def format_trg(self, txt):
      return self.format_h(txt, self.trg_formatter)

    def read_input(self):
      return input()

    def translate(self):
      device = ut.get_device()
      model = Model(self.config).to(device)
      model.load_state_dict(torch.load(self.model_file, map_location=device))
      best_trans_fp = './.interactive.tmp.best_trans'
      beam_trans_fp = './.interactive.tmp.beam_trans'
      input_fp = './.interactive.tmp.input'
      print("Ready")
      txt = self.read_input()
      while txt:
        src_input = self.format_src(txt)
        with open(input_fp, 'w') as inh:
          inh.write(txt + '\n')
        try:
          self.data_manager.translate(model, input_fp, (best_trans_fp, beam_trans_fp), self.logger)
        except:
          print("Error translating")
        with open(best_trans_fp, 'r') as outh:
          output = outh.readline().strip('\n')
        print(output)
        txt = self.read_input()
      self.maybe_terminate_process(self.src_formatter)
      self.maybe_terminate_process(self.trg_formatter)
      os.remove(best_trans_fp)
      os.remove(beam_trans_fp)
      os.remove(input_fp)
