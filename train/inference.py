import pretty_midi
import torch
import librosa
from dataloader.Song import Song
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s
from model.postprocessing import postprocess


def inference(
        audio_path: str,
        return_path: bool | str = False,
        output_path: str = "trial_audio/output.mid",
) -> pretty_midi.PrettyMIDI | str:
    
    device = s.device
    print(f"Evaluating on {device}")
        
    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)
    if s.pre_trained_model_path is not None:
        model.load_state_dict(torch.load(s.pre_trained_model_path, map_location=device))
        print(f"Loaded pre-trained model from {s.pre_trained_model_path}")
    
    model.eval()

    audio, _ = librosa.load(  # type: ignore
            audio_path,
            sr=s.sample_rate,
        )
    audio = torch.tensor(audio).unsqueeze(0)
    audio = audio.to(device)

    if model is HarmonicRNN:
        audio_input = audio.reshape((1, -1, s.sample_rate))
        pred_midi, pred_len, pred_tpb = model(torch.Tensor(audio_input))
        midi = Song.from_np(
            pred_midi[0].to(torch.uint16),
            None,
            int(pred_tpb[0]),
            int(pred_len[0]),
        ).get_midi()

    else:
        with torch.no_grad():
            # Assume audio is already preprocessed and ready for model input
            yo_pred, yp_pred, yn_pred = model(audio)
            
            yo_pred = torch.sigmoid(yo_pred)
            yp_pred = torch.sigmoid(yp_pred)
            yn_pred = torch.sigmoid(yn_pred) if s.remove_yn == False else yp_pred

            yo_pred = yo_pred.squeeze(1) 
            yp_pred = yp_pred.squeeze(1)
            yn_pred = yn_pred.squeeze(1) if s.remove_yn == False else yp_pred

            midi = postprocess(yo_pred, yp_pred, yn_pred, audio_length=audio.shape[-1], sample_rate=s.sample_rate)

        if return_path:
            midi.write(output_path)
            return output_path
        else:
            return midi
