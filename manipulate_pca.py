import sys, os
import numpy as np
import soundfile as sf
from speechbrain.pretrained.interfaces import foreign_class

from sparc import load_model
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem, QMessageBox,
    QLabel, QDoubleSpinBox, QGroupBox, QSpinBox, QSlider, QComboBox
)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# --- NEW: Imports for PCA ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#coder = load_model("multi", device="cuda:0")
coder = load_model("en+", device="cuda:0")



lid = foreign_class(source="TalTechNLP/voxlingua107-xls-r-300m-wav2vec", pymodule_file="encoder_wav2vec_classifier.py", classname="EncoderWav2vecClassifier", hparams_file='inference_wav2vec.yaml', savedir="tmp")
class ArticulatorEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Articulator Trajectory Editor")
        self.setGeometry(100, 100, 1200, 800)

        # --- Data Model ---
        self.original_trajectories = {}
        self.editable_trajectories = {}
        self.static_metadata = {}
        self.param_map = []
        self.current_filepath = None
        self._drag_info = {}
        self.ax_to_map_idx = {}; self.artists = {}

        # --- NEW: PCA Model State ---
        self.pca_scaler = None
        self.pca_model = None
        self.pca_base_transformed = None # The original EMA data in PCA space
        self.pca_sliders = []
        # --- NEW: Speaker Embedding Bank ---
        self.speaker_embeddings = {} # Maps speaker name to spk_emb vector
        self.speaker_pitch_stats = {}
        # --- NEW: State for Blitting Performance
        self.backgrounds = {}

        # --- GUI Elements ---
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(250)

        self.btn_load = QPushButton("Load Trajectories (.npy)")
        self.btn_lid = QPushButton("Identify language")
        self.btn_play = QPushButton("▶ Play Synthesized Audio")
        self.btn_reset = QPushButton("Reset All Trajectories")
        self.btn_dec = QPushButton("Flatten Selected Trajectory")
        self.btn_inc = QPushButton("Exaggerate Selected Trajectory")
        # --- NEW: Speaker Control Group ---
        speaker_group = QGroupBox("Speaker Conversion")
        speaker_layout = QVBoxLayout()
        self.speaker_selector_combo = QComboBox()
        speaker_layout.addWidget(QLabel("Synthesize with Speaker:"))
        speaker_layout.addWidget(self.speaker_selector_combo)
        speaker_group.setLayout(speaker_layout)
        # --- NEW: PCA Control Group ---
        pca_group = QGroupBox("PCA Controls (for EMA)")
        pca_layout = QVBoxLayout()
        self.pca_components_spinner = QSpinBox()
        self.pca_components_spinner.setRange(1, 12); self.pca_components_spinner.setValue(5)
        self.pca_components_spinner.setPrefix("Components: ")
        self.btn_train_pca = QPushButton("Train PCA Model")
        self.pca_sliders_layout = QVBoxLayout() # Will be populated dynamically
        pca_layout.addWidget(self.pca_components_spinner)
        pca_layout.addWidget(self.btn_train_pca)
        pca_layout.addLayout(self.pca_sliders_layout)
        pca_group.setLayout(pca_layout)

        self.articulator_list = QListWidget()
        self.articulator_list.setSelectionMode(QListWidget.ExtendedSelection)
        sculpt_group = QGroupBox("Sculpting Controls")
        sculpt_layout = QVBoxLayout()
        self.radius_spinner = QDoubleSpinBox()
        self.radius_spinner.setRange(1.0, 500.0); self.radius_spinner.setValue(30.0)
        self.radius_spinner.setPrefix("Radius: "); self.radius_spinner.setSingleStep(5.0)
        sculpt_layout.addWidget(self.radius_spinner)
        sculpt_group.setLayout(sculpt_layout)

        left_layout.addWidget(self.btn_load); left_layout.addWidget(self.btn_lid)
        left_layout.addSpacing(10); left_layout.addWidget(self.btn_play)
        left_layout.addSpacing(10); left_layout.addWidget(self.btn_reset)
        left_layout.addSpacing(10); left_layout.addWidget(self.btn_dec)
        left_layout.addSpacing(10); left_layout.addWidget(self.btn_inc)

        
        left_layout.addSpacing(10); left_layout.addWidget(pca_group)
        left_layout.addStretch()

        left_layout.addWidget(speaker_group) # NEW
        left_layout.addSpacing(10); left_layout.addWidget(pca_group)

        
        
        left_layout.addWidget(sculpt_group)
        left_layout.addWidget(QLabel("Parameters:")); left_layout.addWidget(self.articulator_list)
        
        plot_panel = QWidget(); plot_layout = QVBoxLayout(plot_panel)
        self.figure = Figure(); self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar); plot_layout.addWidget(self.canvas)
        main_layout.addWidget(left_panel); main_layout.addWidget(plot_panel)

        # --- Connections ---
        self.btn_load.clicked.connect(self.load_file)
        self.btn_lid.clicked.connect(self.identify_language)
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_reset.clicked.connect(self.reset_all_trajectories)
        self.btn_dec.clicked.connect(self.decrease_trajectory_var)
        self.btn_inc.clicked.connect(self.increase_trajectory_var)
        self.btn_train_pca.clicked.connect(self._train_pca) # NEW
        self.articulator_list.itemSelectionChanged.connect(self.update_plot)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        #self.load_demo_data(); self.update_plot()

      # --- NEW: Helper function for the speaker dropdown ---
    def _update_speaker_selector(self, set_current=None):
        self.speaker_selector_combo.blockSignals(True)
        self.speaker_selector_combo.clear()
        for name in sorted(self.speaker_embeddings.keys()):
            # Store the name as user data for easy retrieval
            self.speaker_selector_combo.addItem(name, userData=name)
        if set_current:
            self.speaker_selector_combo.setCurrentText(set_current)
        self.speaker_selector_combo.blockSignals(False)
    # --- NEW: PCA Functionality ---
    def _train_pca(self):
        if 'ema' not in self.editable_trajectories:
            QMessageBox.warning(self, "Warning", "No 'ema' data found to train PCA model.")
            return

        ema_data = self.editable_trajectories['ema']
        n_components = self.pca_components_spinner.value()

        self.pca_scaler = StandardScaler()
        scaled_data = self.pca_scaler.fit_transform(ema_data)
        self.pca_model = PCA(n_components=n_components)
 
        self.pca_base_transformed = self.pca_model.fit_transform(scaled_data)

        print(f"PCA Trained on {n_components} components.")
        print(f"Total explained variance: {np.sum(self.pca_model.explained_variance_ratio_):.4f}")
        self._setup_pca_sliders()

    def _setup_pca_sliders(self):
        for i in reversed(range(self.pca_sliders_layout.count())): 
            widget = self.pca_sliders_layout.itemAt(i).widget()
            if widget is not None: widget.deleteLater()
        self.pca_sliders = []

        pc_stds = np.std(self.pca_base_transformed, axis=0)

        for i in range(self.pca_model.n_components_):
            label = QLabel(f"PC {i+1} (std: {pc_stds[i]:.2f})")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1000, 1000)
            slider.setValue(0) # Sliders start at zero offset
            
            # Store the scaling factor for this slider
            slider.scale = pc_stds[i] * 5 # Slider range will map to +/- 3 std devs
            
            slider.valueChanged.connect(self._update_from_pca_sliders)
            self.pca_sliders_layout.addWidget(label)
            self.pca_sliders_layout.addWidget(slider)
            self.pca_sliders.append(slider)
 

    def _update_from_pca_sliders_w(self):
        if self.pca_model is None: return

        # 1. Get the per-frame weights (periodicity). Fall back to 1.0 if not available.
        if 'periodicity' in self.editable_trajectories:
            # Shape: (L, 1) to enable broadcasting
            voiced_weight = self.editable_trajectories['periodicity'][:, 0].reshape(-1, 1)
            voiced_weight[voiced_weight<0.5] = 0.5
        else:
            
            num_frames = self.editable_trajectories['ema'].shape[0]
            voiced_weight = np.ones((num_frames, 1))
        
        # 2. Calculate the base offset vector from sliders. Shape: (C,)
        base_offsets = np.zeros(self.pca_model.n_components_)
        for i, slider in enumerate(self.pca_sliders):
            slider_val = slider.value() / 1000.0
            base_offsets[i] = slider_val * slider.scale
        
        # 3. Create the final offset matrix using broadcasting.
        #    (L, 1) * (C,) -> (L, C)
        offset_matrix = voiced_weight * base_offsets
    
        # 4. Additive modification: add the weighted offset matrix
        modified_pca_data = self.pca_base_transformed + offset_matrix[1:,:]
        
        # 5. Inverse transform back to EMA space
        reconstructed_scaled = self.pca_model.inverse_transform(modified_pca_data)
        reconstructed_ema = self.pca_scaler.inverse_transform(reconstructed_scaled)
        
        self.editable_trajectories['ema'] = reconstructed_ema.astype(np.float32)
        self.update_plot() 

    def _update_from_pca_sliders(self):
        if self.pca_model is None: return
    
        offsets = np.zeros(self.pca_model.n_components_)
        for i, slider in enumerate(self.pca_sliders):
            slider_val = slider.value() / 1000.0 # Normalized to [-1, 1]
            offsets[i] = slider_val * slider.scale
            
    
        # Additive modification: add offset to the base PCA trajectories
        modified_pca_data = self.pca_base_transformed + offsets
        
        
        reconstructed_scaled = self.pca_model.inverse_transform(modified_pca_data)
        reconstructed_ema = self.pca_scaler.inverse_transform(reconstructed_scaled)
        
        self.editable_trajectories['ema'] = reconstructed_ema.astype(np.float32)
        self.update_plot()

    def reset_all_trajectories(self):
        self.editable_trajectories = {k: v.copy() for k, v in self.original_trajectories.items()}
        # Clear PCA model and sliders
        self.pca_model = None
        self.pca_scaler = None
        self.pca_base_transformed = None
        for i in reversed(range(self.pca_sliders_layout.count())): 
            widget = self.pca_sliders_layout.itemAt(i).widget()
            if widget is not None: widget.deleteLater()
        self.pca_sliders = []
        self.update_plot()

      # --- Data Handling (MODIFIED) ---
    def _setup_data(self, data_dict, speaker_name="Unknown"):
        self.reset_all_trajectories()
        self.original_trajectories = {}
        self.static_metadata = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.ndim == 2: self.original_trajectories[key] = value
            else: self.static_metadata[key] = value
        self.editable_trajectories = {k: v.copy() for k, v in self.original_trajectories.items()}
        print (self.static_metadata)
        # --- NEW: Capture Speaker Embedding ---
        if 'spk_emb' in self.static_metadata:
            self.speaker_embeddings[speaker_name] = self.static_metadata['spk_emb']
            self.speaker_pitch_stats[speaker_name] = self.static_metadata['pitch_stats']
            self._update_speaker_selector(set_current=speaker_name)
            print(f"Captured speaker embedding for: {speaker_name}")
        
        print(f"Loaded editable trajectories: {list(self.original_trajectories.keys())}")
        print(f"Loaded static metadata: {list(self.static_metadata.keys())}")
        self.param_map = []
        ema_labels = ['TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY']
        key_order = ['pitch', 'loudness', 'ema', 'periodicity']
        # ... (rest of the method is the same)
        for key in key_order:
            if key in self.original_trajectories:
                if key == 'ema':
                    for i in range(self.original_trajectories[key].shape[1]):
                        label = ema_labels[i] if i < len(ema_labels) else f"EMA_{i+1}"
                        self.param_map.append({'key': key, 'col': i, 'name': label})
                else: self.param_map.append({'key': key, 'col': 0, 'name': key})
        for key, value in self.original_trajectories.items():
            if key not in key_order:
                 for i in range(value.shape[1]): self.param_map.append({'key': key, 'col': i, 'name': f"{key}_{i+1}"})
        self.populate_articulator_list()


    def load_wav(self,filepath):

        param_dict = coder.encode(filepath) #"test_wavs/fact_color_0014.wav")
        self._setup_data(param_dict, speaker_name=os.path.basename(filepath).split('.')[0])

    def load_file(self):

        filepath, _ = QFileDialog.getOpenFileName(self, "Open", "", "Audio files (*.wav)")
        if filepath:
            try:
                self.load_wav(filepath)
                self.current_filepath = filepath
                self.setWindowTitle(f"Editor - {filepath}")
                self.update_plot()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}\n\nMust be a .wav file")
                
    def identify_language(self, top_k=5):
        import torch

        wav = self.synthesize()
        sf.write('tmp_resynth.wav', wav, 16000)
        top_k = 5
        out_prob, score, index, text_lab = lid.classify_file("tmp_resynth.wav")
        log_likelihoods = out_prob.squeeze()
        probabilities = torch.nn.functional.softmax(log_likelihoods, dim=-1)
        top_k_probabilities, top_k_indices = torch.topk(probabilities, top_k)
        # Get the label encoder to map indices to language labels
        label_encoder = lid.hparams.label_encoder

        # Print the top-k predictions
        print(f"Top-{top_k} language predictions:")
        for i in range(top_k):
            language = label_encoder.decode_torch(top_k_indices[i].unsqueeze(0))
            probability = top_k_probabilities[i].item()
            print(f"- {language[0]}: {probability:.4f}")
            #print(text_lab, score.exp())
        
    def save_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save", "", "NumPy files (*.npy)")
        if filepath:
            try:
                full_dict_to_save = self.editable_trajectories.copy()
                full_dict_to_save.update(self.static_metadata)
                np.save(filepath, full_dict_to_save)
                print(f"Saved full data dictionary to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def _apply_speaker_pitch_range(self, f0, tgt_mean, tgt_std):
        return f0-np.mean(f0)+tgt_mean
        norm = (f0 - np.mean(f0)) / np.std(f0)
        return (norm * tgt_std)+tgt_mean

    def synthesize(self):
        if self.editable_trajectories:
            #self.btn_play.setText("Synthesizing...")
            QApplication.processEvents()
            try:
                full_dict_for_synth = self.editable_trajectories.copy()
                full_dict_for_synth.update(self.static_metadata)

                # --- NEW: Override speaker embedding based on user selection ---
                selected_speaker = self.speaker_selector_combo.currentData()
                if selected_speaker and selected_speaker in self.speaker_embeddings:
                    print(f"Synthesizing with selected speaker: {selected_speaker}")
                    full_dict_for_synth['spk_emb'] = self.speaker_embeddings[selected_speaker]
                    mean, std =  self.speaker_pitch_stats[selected_speaker]
                    
                    full_dict_for_synth['pitch'] = self._apply_speaker_pitch_range(full_dict_for_synth['pitch'], mean, std)
                    
                else:
                    print("Synthesizing with original speaker.")

                wav = coder.decode(**full_dict_for_synth)
                return wav
            except Exception as e: QMessageBox.critical(self, "Synthesis Error", f"An error occurred: {e}")

    def play_audio(self):
        wav = self.synthesize()
        sf.write('ppa_resynth.wav', wav, 16000)
        self.btn_play.setText("▶ Play Synthesized Audio")
        os.system("play ppa_resynth.wav trim 0 5  &")
        
    def update_plot(self):
        print("updating")
        if not self.editable_trajectories: return
        xlim = self.figure.get_axes()[0].get_xlim() if self.figure.get_axes() else None
        self.figure.clear(); self.artists = {}; self.ax_to_map_idx = {}
        selected_map_indices = self._get_selected_map_indices()
        if not selected_map_indices: self.canvas.draw(); return
        axes = self.figure.subplots(len(selected_map_indices), 1, sharex=True)
        if len(selected_map_indices) == 1: axes = [axes]
        for i, map_idx in enumerate(sorted(list(selected_map_indices))):
            ax = axes[i]; self.ax_to_map_idx[ax] = map_idx
            param_info = self.param_map[map_idx]
            key, col, name = param_info['key'], param_info['col'], param_info['name']
            orig_data = self.original_trajectories[key][:, col]
            edit_data = self.editable_trajectories[key][:, col]
            orig_line, = ax.plot(orig_data, '--', color='gray', alpha=0.7, zorder=1)
            edit_line, = ax.plot(edit_data, zorder=2)
            self.artists[map_idx] = {'orig': orig_line, 'edit': edit_line, 'ax': ax}
            ax.set_ylabel(name); ax.grid(True)
        axes[-1].set_xlabel("Time (frames)")
        if xlim: axes[0].set_xlim(xlim)
        self.figure.tight_layout(); self.figure.subplots_adjust(hspace=0.1)
        self.canvas.draw()

    def on_press(self, event):
        if self.toolbar.mode or event.button != 1: return
        map_idx = self.ax_to_map_idx.get(event.inaxes)
        if map_idx is None or event.xdata is None: return
        param_info = self.param_map[map_idx]
        key, col = param_info['key'], param_info['col']
        self._drag_info = {
            'map_idx': map_idx, 'start_x': event.xdata, 'start_y': event.ydata,
            'trajectory_at_start': self.editable_trajectories[key][:, col].copy()
        }


    def on_motion(self, event):

        if not self._drag_info or event.inaxes is None or event.ydata is None: return
        map_idx = self._drag_info['map_idx']
        if self.ax_to_map_idx.get(event.inaxes) != map_idx: return
        param_info = self.param_map[map_idx]
        key, col = param_info['key'], param_info['col']
        num_frames = self.editable_trajectories[key].shape[0]
        x_indices = np.arange(num_frames)
        radius = self.radius_spinner.value()
        sigma = radius / 3.0
        center_x = self._drag_info['start_x']
        gaussian = np.exp(-0.5 * ((x_indices - center_x) / sigma) ** 2)
        delta_y = event.ydata - self._drag_info['start_y']
        deformation = delta_y * gaussian
        self.editable_trajectories[key][:, col] = self._drag_info['trajectory_at_start'] + deformation
        self.artists[map_idx]['edit'].set_ydata(self.editable_trajectories[key][:, col])

        self.canvas.draw_idle()

    def on_release(self, event):
        #self.canvas.draw_idle()
        self._drag_info = {}
    
    def _get_selected_map_indices(self): return {self.articulator_list.item(i).data(Qt.UserRole) for i in range(self.articulator_list.count()) if self.articulator_list.item(i).isSelected()}
    
    def populate_articulator_list(self):
        self.articulator_list.blockSignals(True)
        self.articulator_list.clear()

        for i, param_info in enumerate(self.param_map):
            item = QListWidgetItem(param_info['name'])
            item.setData(Qt.UserRole, i)
            self.articulator_list.addItem(item)
            item.setSelected(True)
        self.articulator_list.blockSignals(False)

    def reset_selected_trajectories(self):
        # This now only resets non-EMA channels, use the global reset for everything
        if not self.original_trajectories: return
        selected_map_indices = self._get_selected_map_indices()
        if not selected_map_indices: QMessageBox.information(self, "Info", "Select parameters to reset."); return
        for map_idx in selected_map_indices:
            param_info = self.param_map[map_idx]
            key, col = param_info['key'], param_info['col']
            # Only reset non-ema, as ema is controlled by PCA or global reset
            if key != 'ema':
                self.editable_trajectories[key][:, col] = self.original_trajectories[key][:, col]
        self.update_plot()

    def increase_trajectory_var(self):
        if not self.editable_trajectories: return
        selected_map_indices = self._get_selected_map_indices()
        if not selected_map_indices: QMessageBox.information(self, "Info", "Select parameters to exaggerate."); return
        for map_idx in selected_map_indices:
            param_info = self.param_map[map_idx]
            key, col = param_info['key'], param_info['col']
            
            trajectory = self.editable_trajectories[key][:, col]
            mean_val = np.mean(trajectory)
            self.editable_trajectories[key][:, col] = mean_val + 1.2 * (trajectory - mean_val)
        self.update_plot()

    def decrease_trajectory_var(self):
        if not self.editable_trajectories: return
        selected_map_indices = self._get_selected_map_indices()
        if not selected_map_indices: QMessageBox.information(self, "Info", "Select parameters to flatten."); return
        for map_idx in selected_map_indices:
            param_info = self.param_map[map_idx]
            key, col = param_info['key'], param_info['col']

            trajectory = self.editable_trajectories[key][:, col]
            mean_val = np.mean(trajectory)
            self.editable_trajectories[key][:, col] = mean_val + 0.8 * (trajectory - mean_val)
        self.update_plot()

if __name__ == '__main__':
    # Create a dummy test_wavs directory if it doesn't exist
    if not os.path.exists("test_wavs"):
        os.makedirs("test_wavs")
        print("Created dummy 'test_wavs' directory. Please add a wav file like 'LJ001-0001_1.wav' to it.")

    app = QApplication(sys.argv)
    editor = ArticulatorEditor()
    editor.show()
    sys.exit(app.exec())
