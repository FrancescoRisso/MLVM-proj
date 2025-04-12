# Appunti MIDI

## Libreria mido

La libreria [mido](https://mido.readthedocs.io/en/stable/index.html]) è molto fornita per quanto riguarda i midi.

## Formato midi

Midi è un formato audio basato su messaggi.
Normalmente c'è una periferica esterna (es pianoforte) che manda questi messaggi al sampler, che li salva come una sequenza.
Ci sono tuttavia dei [meta-messaggi](https://www.recordingblogs.com/wiki/midi-meta-messages), che sono salvati nei file .midi ma non mandati da periferiche esterne.

### Tick

Il formato midi conosce il tempo in maniera discreta, sotto forma di tick.

Ogni file ha un tempo, salvato come meta messaggio in una delle track.
Il tempo è la durata, in μs, di una nota da 1/4 (semiminima): indica quindi la velocità a cui la canzone è suonata.
Sovente è espresso in un'altra unità di misura, BPM (battute al minuto)

Per ogni canzone si può poi definire un tick, come una frazione della durata di una semiminima.
Solitamente una semiminima è divisa in 12, 24, 48 o 96 tick, in base a quanta risoluzione si vuole avere.
Per esempio, se la semiminima vale 12 tick, posso avere nella canzone crome (6 tick) e semicrome (3 tick), ma nulla di più piccolo (il fattore 3 serve per le triadi).
Ogni canzone ha un certo numero di tick per battuta, che oltre a definire questa precisione, definisce anche in un certo senso il tempo della canzone (es con semiminima = 12 tick, una canzone in 4/4 avrà 48 tick per battuta, una canzone in 3/4 ne avrà solo 36).
Il tick è quindi un'indicazione della durata relativa delle note all'interno della stessa canzone, per esempio una nota da 3n tick durerà sempre 3 volte una nota da n tick, indipendentemente dal tempo.
Il valore di tick per battuta è salvato direttamente nel file

La durata reale di un tick dipende da tick per battuta e tempo, cioè dipende dalla durata relativa della nota e da quanto allungo tutte le note.

### Track

Un midi è composto da N track, che idealmente sono in rapporto 1:1 con gli strumenti registrati.
In realtà:

-   a volte si smezza uno strumento su più track per avere più canali (vedi [sotto](#canali))
-   a volte sembra si usi una track separata per salvare alcuni metadati, tipo il tempo, il copyright, ... della canzone

Nel nostro caso, le tracce sono composte di due track, di cui la prima contiene solo metadati, mentre la seconda contiene tutto quello suonato dal piano.

### Canali

Ogni track può essere composta di 16 canali (solitamente numerati 1-16, ma nella libreria che ho trovato sono numerati 0-15).
Ogni canale idealmente contiene una parte diversa dello strumento, per produrre suoni diversi.
Per esempio, gli organi delle chiese hanno N tastiere + la pedaliera, quindi indicativamente usano N+1 canali, che possono essere riprodotti con font musicali diversi (es tastiera1 suona con il font audio del pianoforte, tastiera2 suona con il font della chitarra).
Strumenti invece tipo la chitarra hanno solo una possibilità di suono, quindi hanno un solo canale.

Nel nostro caso, è tutto registrato sul canale 0

### Messaggi

Come si diceva, i dati sono salvati come messaggio. Ogni messaggio ha:

-   tipo
-   time, che indica il tempo in tick trascorso dal messaggio precedente, potenzialmente anche 0 (quindi per sapere a che tick c'è un messaggio, devo fare la somma dei time di tutti i messaggi precedenti)
-   _altri parametri_

#### Messaggi note

Una nota inizia con un messaggio con tipo "note_on", e finisce con un messaggio con tipo "note_off" (oppure un messaggio "note_on" con velocity = 0).

Questi tipi di messaggio hanno come campi:

-   channel
-   note: l'effettiva nota che è suonata. Il do centrale è note=60, poi ogni semitono in salita o in discesa aumenta o diminuisce di 1 il valore di note. Il range è 0-127 (inclusi)
-   velocity: quanto forte la nota è suonata, su un range 0-127 (inclusi). Come detto, prima un "note_on" con velocity=0 è equivalente ad un "note_off"
-   (time)

#### Messaggi di controllo

Sono messaggi con tipo "control_change".

In particolare hanno:

-   channel
-   control: che cosa vanno a modificare (es volume, effetti, ... [qui](https://nickfever.com/music/midi-cc-list) c'è la lista completa)
-   value: il valore che settano per il campo di controllo (es control=_volume_ ==> value = a quanto settare il volume)
-   (time)
