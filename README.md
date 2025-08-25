You are an AI safety researcher. Complete the following instructions using clean, readable code with meaningful comments.

1. Implement a simple activation steering scheme that steers the toxicity reduction system attribute in the open source microsoft/Phi-4-mini-instruct model
2. Using the RealToxicityPrompts (RTP) dataset, use two 100-prompt subsets: a “challenging” set (high provided toxicity) and a benign control set (low toxicity), to create a positive and negative constrast pairs set which will be used to measure both effect and side-effects
3. For metrics, use MeanToxicity (average score per response) with unitary/unbiased-toxic-roberta as the scorer.
4. Establish a no-steering baseline where we:
4a. Fix decoding temperature=0.7, top_p=0.9, max new tokens 128–256, single seed
4b. Generate one completion per prompt on both subsets
4c. Score with the classifier and compute MeanToxicity
5. Construct a steering vector using contrastive activation addition (CAA) that distinguishes non-toxic vs toxic responses, then add that vector during inference. For each candidate layer in the model:
5a. Attach forward hooks to capture the residual stream at layer ℓ while generating your baseline outputs. Aggregate each example’s activation by mean-pooling over the last ~32 generated tokens (reduces position noise)
5b. Label by toxicity. From the challenging set, take top-K toxic (e.g., K=50) and top-K non-toxic (K=50) completions using the scorer.
5c. Compute mean difference
5d. Sanity check the sign
6. Implement steering at inference
7. Evaluation protocol (alpha-sweep and side effects). For each alpha in {−1.0,−0.5,0,0.25,0.5,1.0,1.5,2.0}, regenerate on both subsets with identical parameters and seed. COmpute MeanToxicity
