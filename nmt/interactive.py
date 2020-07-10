import os
import torch
import subprocess
import time
import sys
import tempfile
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations

class InteractiveTranslator(object):
    def __init__(self, args):
        super(InteractiveTranslator, self).__init__()
        self.config = configurations.get_config(args.proto, getattr(configurations, args.proto), args.config_overrides)
        
        self.model_file = args.model_file
        if self.model_file is None:
            self.model_file = os.path.join(self.config['save_to'], '{}.pth'.format(self.config['model_name']))

        if not os.path.exists(self.model_file):
            raise ValueError('Model file does not exist')

        self.data_manager = DataManager(self.config)

        _, self.input_fp = tempfile.mkstemp()
        _, self.output_fp = tempfile.mkstemp()
        self.input_fhw = open(self.input_fp, 'wb', buffering=1)
        self.input_fhr = open(self.input_fp, 'r', buffering=1)
        self.output_fhw = open(self.output_fp, 'wb', buffering=1)
        self.output_fhr = open(self.output_fp, 'r', buffering=1)

        self.preprocessor = self.maybe_spawn_process(args.preprocessor, self.input_fhw)
        self.postprocessor = self.maybe_spawn_process(args.postprocessor, self.output_fhw)

        self.translate()

    def maybe_spawn_process(self, cmd, outh):
        if cmd:
            return subprocess.Popen(cmd, bufsize=1, stdin=subprocess.PIPE, stdout=outh, stderr=subprocess.STDOUT)

    def communicate_with_process(self, process, msg, fh):
        if process:
            msg = msg.replace('\n', '\\n') + '\n'
            process.stdin.write(msg.encode('utf-8'))
            process.stdin.flush()
            out = None
            while not out:
                time.sleep(0.01)
                out = fh.readline()
            return out.rstrip('\n').replace('\\n', '\n')
        else:
            return msg

    def terminate(self):
        if self.preprocessor: self.preprocessor.terminate()
        if self.postprocessor: self.postprocessor.terminate()
        self.input_fhw.close()
        self.input_fhr.close()
        self.output_fhw.close()
        self.output_fhr.close()
        os.remove(self.input_fp)
        os.remove(self.output_fp)

    def translate(self):
        device = ut.get_device()
        model = Model(self.config).to(device)
        model.load_state_dict(torch.load(self.model_file, map_location=device))
        for line in sys.stdin:
            txt = line.rstrip('\n')
            src_input = self.communicate_with_process(self.preprocessor, txt, self.input_fhr)
            try: trans = self.data_manager.translate_line(model, src_input)
            except: trans = ""
            print(self.communicate_with_process(self.postprocessor, trans, self.output_fhr))
        self.terminate()
