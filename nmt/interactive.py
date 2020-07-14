import os
import torch
import subprocess
import time
import sys
import tempfile
import nmt.utils as ut
import nmt.all_constants as ac
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations

class InteractiveTranslator(object):
    def __init__(self, args):
        super(InteractiveTranslator, self).__init__()
        self.minibatch = args.minibatch_interactive
        self.config = configurations.get_config(args.proto, getattr(configurations, args.proto), args.config_overrides)

        self.batch_size = self.config['batch_size'] if args.minibatch_interactive else None
        self.struct = self.config['struct']

        self.new_batch()
        
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

        device = ut.get_device()
        self.model = Model(self.config).to(device)
        self.model.load_state_dict(torch.load(self.model_file, map_location=device))
        self.model.eval()

        self.read()

    def new_batch(self):
        self.current_batch_toks = []
        self.current_batch_structs = []
        self.current_batch_size = 0

    def add_to_batch(self, line):
        try: prsd = self.struct.parse(line)
        except AssertionError: pass
        prsd = prsd.map(lambda w: self.data_manager.src_vocab.get(w, ac.UNK_ID))
        prsd.maybe_add_eos(ac.EOS_ID)
        size = prsd.size()

        if self.batch_size and self.current_batch_size + size > self.batch_size:
            self.translate()
            self.new_batch()
        self.current_batch_toks.append(torch.tensor(prsd.flatten()))
        self.current_batch_structs.append(prsd)
        self.current_batch_size += size
        if not self.batch_size: # if no buffering
            self.translate()
            self.new_batch()

    def translate(self):
        toks = torch.nn.utils.rnn.pad_sequence(self.current_batch_toks, padding_value=ac.PAD_ID, batch_first=True)
        for line in self.data_manager.translate_batch(self.model, toks, self.current_batch_structs):
            print(self.communicate_with_process(self.postprocessor, line, self.output_fhr))

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

    def read(self):
        for line in sys.stdin:
            txt = line.rstrip('\n')
            src_input = self.communicate_with_process(self.preprocessor, txt, self.input_fhr)
            #try: trans = self.data_manager.translate_line(model, src_input)
            #except: trans = ""
            #print(self.communicate_with_process(self.postprocessor, trans, self.output_fhr))
            self.add_to_batch(src_input)
        self.terminate()
