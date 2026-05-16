import pickle
for f in ['gemini_results.pkl', 'lora_results.pkl']:
    with open(f, 'rb') as fp:
        d = pickle.load(fp)
    print(f, ': n_samples =', d['n_samples'], '| predictions len =', len(d['predictions']))