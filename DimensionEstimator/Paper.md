# Correlation Dimension of Natural Language in a Statistical Manifold 

Xin Du 1, ∗ and Kumiko Tanaka-Ishii 2, †

> 1

Waseda Research Institute for Science and Engineering, Waseda University. 

> 2

Department of Computer Science and Engineering, School of Fundamental Science and Engineering, Waseda University. 

(Dated: May 16, 2024) 

The correlation dimension of natural language is measured by applying the Grassberger-Procaccia algorithm to high-dimensional sequences produced by a large-scale language model. This method, previously studied only in a Euclidean space, is reformulated in a statistical manifold via the Fisher-Rao distance. Language exhibits a multifractal, with global self-similarity and a universal dimension around 6.5, which is smaller than those of simple discrete random sequences and larger than that of a Barab´ asi-Albert process. Long memory is the key to producing self-similarity. Our method is applicable to any probabilistic model of real-world discrete sequences, and we show an application to music data. 

CONTENTS 

I. Introduction 2II. Method 2III. Results 4Acknowledgments 8References 8A. Properties of the Mapping ϕ : xt 7 → pt 91. Formulation 92. Linearity 93. Distance Distortion Rate 94. Dimension Preservation for Markov Processes 10 a. When a≥t and a≥s Follow the Same Markov Process 10 b. When a≥t and a≥s Follow Different Markov Processes 11 B. GPT-Like Large-Scale Language Models 14 C. Dimension Reduction 15 D. Local Fractality 16 1. Comparison with Dirichlet Distribution 16 2. Local Fractals under Small Context Length 18 E. Our Data 20 F. Supplementary Results on Book Texts 21 1. Effect of Sequence Length 21 2. Effect of Context Length 21 G. Comparison with Random Processes 22 1. Uniform White Noise 22 2. Symmetric Dirichlet White Noise 23 3. Barab´ asi-Albert Network and A Fractional Variant 23 H. Using Euclidean Distance Metric 26 I. Correlation dimension of music data 28  

> ∗

duxin.ac@gmail.com † kumiko@waseda.jp 

> arXiv:2405.06321v2 [cs.CL] 15 May 2024

2

> I. INTRODUCTION

The correlation dimension of Grassberger and Procac-cia (1983) quantifies the degree of recurrence in a sys-tem’s evolution and has been applied to examine the characteristics of sequential data, such as the trajectories of strange attractors (Grassberger and Procaccia, 1983), random processes (Osborne and Provenzale, 1989), and sequences sampled from complex networks (Lacasa and G´ omez-Gardenes, 2013). In this letter, we report the correlation dimension of natural language by regarding texts as the trajectories of a language dynamical system. In contrast to the long-memory quality of natural language as reported in (Alt-mann et al. , 2012; Li, 1989; Tanaka-Ishii and Bunde, 2016), the correlation dimension of natural language has barely been studied because of its high dimensionality and discrete nature. An exceptional previous work, to the best of our knowledge, was that of Doxas et al. 

(2010), who measured the correlation dimension of lan-guage in terms of a set of paragraphs. Every paragraph was represented as a vector, with each dimension being the logarithm of a word’s frequency. The distance be-tween two paragraphs was measured as the Euclidean distance. Such a representation has also been used for measuring other scaling factors of language (Ausloos, 2012; Kobayashi and Tanaka-Ishii, 2018; Tanaka-Ishii and Kobayashi, 2018). However, without a rigorous defi-nition of language as a dynamical system, the correlation dimension is difficult to interpret, and its value may eas-ily depend on the setting. For example, the dimension would vary greatly between handling word frequencies logarithmically and nonlogarithmically. Today, language representation has become elaborate by incorporating semantic ambiguity and long context. 

Large language models (LLMs) (OpenAI, 2023; Radford 

et al. , 2019; Touvron et al. , 2023; Yi, 2024) such as Chat-GPT generate texts that are hardly distinguishable from human-generated texts. The generation process is au-toregressive, which naturally associates a dynamical sys-tem. Such state-of-the-art (SOTA) models (i.e., the GPT series, including GPT-4 (OpenAI, 2023), Llama-2 (Tou-vron et al. , 2023), and “Yi” (Yi, 2024)) have opened a new possibility of studying the physical nature of lan-guage as a complex dynamical system. Furthermore, ex-ploration of the fractal dimension of language offers a novel approach to examine the underlying structures of pretrained neural networks, thus shedding light on the intricate ways they mirror human intelligence. These new systems, however, are not defined in a Eu-clidean space and thus require reformulation of the state space and the metric between states. Because a neural model assumes a probability space, the analysis method that was originally defined in a Euclidean space must be accommodated in a space of probability distributions, and the distance metric must be statistical. Specifi-cally, we consider a statistical manifold (Amari, 2012; Rao, 1992) whose metric is the Fisher information met-ric. Hence, this letter proposes a rigorous formalization to analyze the universal properties of these GPT mod-els, thus representing language as an original dynamical system. Although we report results mainly for language, given the impact of ChatGPT, our formalization applies to any other GPT neural models for real-world sequences, such as DNA, music, programming sources, and finance data. To demonstrate this possibility, we show an appli-cation to music. 

> II. METHOD

Let ( S, d ) be a metric space and [ x1, x 2, · · · , x N ] be a point sequence, where xt ∈ S for t = 1 , · · · , N . The Grassberger-Procaccia algorithm (Grassberger and Pro-caccia, 1983) (GP in the following) defines the correlation dimension of this point sequence in terms of an exponent 

ν via the growth of the correlation integral C(ε), as fol-lows: 

C(ε) ∼ εν as ε → 0, (1) where 

C(ε) = lim  

> N→∞

1

N 2

X

> 1≤t,s ≤N

#

n

(t, s ) : d(xt, x s) < ε 

o

, (2) # denotes a set’s size, and d is the distance metric. In the original GP, the sequence lies in a Euclidean space and d is the Euclidean distance. For an ergodic se-quence, the correlation dimension suggests the values of other fractal dimensions such as the Haussdorf dimen-sion (Pesin, 1993). For example, the H´ enon map has 

ν = 1 .21 ±0.01 (Grassberger and Procaccia, 1983), which is close to its Hausdorff dimension of 1 .261 ± 0.003 (Rus-sell et al. , 1980). GP can be generalized to apply to a sequence in a more general smooth manifold (Pesin, 1993). In our study, we examine natural language through this correlation dimension. Thus far, language texts have typically been considered in a Euclidean space. How-ever, recent large language models have shown unprece-dented performance in the form of an autoregressive sys-tem, which is defined in a probability space. Hence, we are motivated to measure the correlation dimension in a statistical manifold. We consider a language dynamical system {xt} that develops word by word: f : xt 7 → xt+1 . Let V rep-resent a vocabulary that comprises all unique words. A sequence of words, a = [a1, a 2, · · · , a t, · · · ], where 

at ∈ V , is associated with a sequence of system states, [x1, x 2, · · · , x t, · · · ]. As demonstrated in Figure 1(a) at the top, we define each state xt as a probability distribu-tion over the set Γ of all word sequences. xt measures the 3                      

> (a) (b) FIG. 1 Our model of language as a stochastic dynamical system. (a) The difference between the system state xtand the next-word probability distribution pt.(b) {pt}(where pt∈Mult( V)) as the image of {xt}(where xt∈S) through the marginalization mapping ϕin Formula (5). In this study, we use ˆ νto approximate ν.

probability of any text to occur as a≥t = [ at, a t+1 , · · · ], following a context a<t = [ a1, · · · , a t−1]. Furthermore, we consider the next-word probability distribution pt

over the vocabulary V . xt and pt are formally defined as follows: 

xt(a≥t) = P( a≥t | a<t ) ∀a≥t ∈ Γ, (3) 

pt(w) = P( at = w | a<t ) ∀w ∈ V. (4) 

pt can be represented as the image of xt by a mapping ϕ:

pt = ϕ(xt). (5) Here, ϕ is the marginalization across Γ and is linear with respect to a mixture of distributions, as explained in Supp. A.2. Hence, a language state xt is represented as a probabil-ity function instead of a point in a Euclidean space. The correlation dimension ν can be defined for the sequence 

{xt} as long as the distance metric d in Formula (2) is specified between any pair of states xt and xs. However, direct acquisition of d(xt, x s) is nontrivial because {xt}

as a language is unobservable. One new alternative path today is to represent xt via pt, where pt is produced by a large language (especially a GPT-like) model (LLM). We denote the correlation dimension of the sequence {pt} as ˆν. Our approach is summarized in Figure 1(b) at the bot-tom. Supp. B provides a brief introduction to GPT-like LLMs. Theoretically, ˆ ν = ν when the sequence of words is generated by a Markov process. We prove this in Supp. A.4. Natural language exhibits the Markov property to a certain extent, but strictly speaking, it violates the prop-erty. This phenomenon has been studied in terms of long memory (Altmann et al. , 2012, 2009; Li, 1989; Tanaka-Ishii and Bunde, 2016), as mentioned in the Introduction. Therefore, the ˆ ν acquired from pt will remain an approx-imation of ν. In general, ˆ ν ≤ ν holds (Peitgen et al. ,1992) and ˆ ν thus constitutes a lower bound of ν.The distance metric d in Formula (2) is chosen as the Fisher-Rao distance, defined as the geodesic distance on a statistical manifold generated by Fisher information (Amari, 2012). When {pt} is presumed to follow a multi-noulli distribution (over the vocabulary V ), the statisti-cal manifold is the space of all multinoulli distributions over V , denoted as Mult( V ), as shown at the top right in Figure 1(b). Mult( V ) has a (topological) dimension of 

|V | − 1 and is isometric to the positive orthant of a hy-persphere. The Fisher-Rao distance is analytically equal to twice the Bhattacharyya angle, as follows: 

dFR (pt, p s) = 2 arccos X

> w∈V

ppt(w)ps(w)

!

for t, s = 1 , 2, · · · , N. 

(6) This statistical manifold is a Riemannian manifold of constant curvature (as it constitutes a part of a hyper-sphere), sharing many favorable topological properties with Euclidean spaces. Particularly, the Marstrand pro-jection theorems (Falconer, 2004; Marstrand, 1954) for Euclidean spaces, which state that linear mappings al-most surely preserve a set’s Hausdorff dimension, can be generalized to such Riemannian manifolds. Recently, Balogh and Iseli (2016) proved Marstrand-like theorems for sets on a 2-sphere. Because the mapping ϕ : xt 7 → pt

is linear, as mentioned before and proved in Supp. A.2, these theorems could be generalized to suggest the equal-ity ν = ˆ ν. This possible generalization goes beyond this letter’s scope; even if it were true, Marstrand-like the-orems do not guarantee a specific linear mapping (i.e., 

ϕ) to be dimension-preserving. Nevertheless, these the-orems motivate our proposal to analyze ν via its lower bound ˆ ν.The calculation of distances over N timesteps takes 

O(|V | · N 2) time, with a vocabulary size |V | around 10 4.This computational cost can be reduced to O(M · N 2)through dimension reduction from {pt} to {qt}, without altering the estimated correlation dimension ˆ ν, where 4

M ≪ | V | is the new, smaller dimensionality. For t =1, · · · , N , the dimension-reduction projection transforms 

pt to qt as follows: 

qt(m) = X

> w∈Φ−1({m})

pt(w), ∀m = 1 , · · · , M. (7) Here, Φ is determined via the modulo function: Φ( w) = index( w) mod M , where index( w) indicates a word’s in-dex in the vocabulary. Essentially, we “randomly” group words from the extensive vocabulary V in a smaller set 

{1, · · · , M } and estimate ˆ ν according to this condensed vocabulary. We empirically validated this method, which is rooted in Marstand’s projection theorem, as detailed in Supp. C. Specifically, dimensionality reduction from approximately 50,000 to 1,000 retained the consistency of estimating ˆ ν and achieved up to 50X faster computation. 

> III. RESULTS

Before showing the correlation dimension, we examine language’s inherent self-similarity. Figure 2 includes a plot showing the probability pt of encountering “,” (com-mas) and “;” semicolons over t = 1 , 2, · · · , N in an En-glish translation of Don Quixote by Miguel de Cervantes from Project Gutenberg 1. These punctuation marks, chosen for their high frequency, illustrate the role of se-mantic ambiguity. Each pt represents a point in Mult( V ), a probability vector of the next-word occurrence, esti-mated using GPT2-xl (Radford et al. , 2019). The figure maps these points, varying with input context a<t , and classifies them by Shannon entropy H(pt), revealing self-similarity in both low- and high-entropy regions through magnified views at different scales. Nevertheless, a thor-ough assessment of this self-similarity necessitates exam-ining the high-dimensional space of Mult( V ), beyond the limits of a two-dimensional display that cannot represent correlation dimensions above 2. We conjecture that the trajectory has two kinds of frac-tals: local and global. The local fractals, potentially aris-ing from simple word distributions across contexts akin to those in topic models like LDA (Blei et al. , 2003), are evident in low-entropy areas where single words pre-dominate. In Supp. D.1, we show that even i.i.d. sam-ples from a Dirichlet distribution (a commonly assumed prior for multinoulli distributions) can reproduce the lo-cal fractal seen in Figure 2. The local kind’s occurrence could be related to the finding in Doxas et al. (2010) that topic models can reproduce self-similar patterns. How-ever, the local kind is not especially concerned in this let-ter because it characterizes single words and hence does not reveal the nature of the original system {xt}. 

> 1https://www.gutenberg.org/ebooks/996

In this letter, we are mainly interested in the corre-lation dimension of the global phenomenon. Unlike the local kind, the global fractals represent high-entropy re-gions that are governed by the trajectory’s global devel-opment. Hence, we consider points in the higher-entropy region, as filtered by a parameter η:max  

> w∈V

pt(w) < η. (8) Figure 3(a) shows the correlation integral from For-mula (2) with respect to ε for Don Quixote in terms of different probability thresholds η in Formula (8). As η

decreases to 0.5 (red curve), the linear region becomes visible across all scales, and the correlation dimension (given by the slope) converges to ˆ ν = 6 .42. In contrast, the curve for η = 1 .0 (i.e., when no timesteps are ex-cluded) shows great deviation from the other curves, es-pecially at smaller ε values, producing a local correlation dimension that drops below 2.0. Hence, unless mentioned otherwise, η = 0 .5 in this letter. For η = 0 .5, Figure 3(a) shows a long span across more than six orders of magni-tude, from 10 −1 to 10 −8 on the vertical axis. Figure 3(b) characterizes the effect of N , the length of the text used to estimate the correlation dimension. The longest text fragment had 150,000 words and is in-dicated by the red curve. Convergence is visible for all 

N , starting from N = 500. Unless mentioned otherwise, 

N = 150 , 000 here. We also investigated the effect of the context length, denoted as c. Ideally, an LLM estimates the distribution 

pt by using the whole text [ a1, · · · , a t−1] before timestep 

t as the context, but in practice, a maximum context length c is often set; that is, 

p(c) 

> t

(w) = P( at = w | at−c, a t−c+1 , · · · , a t−1)

≈ pt(w) ∀w ∈ V. (9) Unless mentioned otherwise, all results in this letter were obtained with c = 512. Figure 3(c) shows the correlation dimension with val-ues of c as small as 1 (i.e., a Markov model). For context lengths above 32, a clear linear scaling phenomenon is observed across all scales, which resembles the case of 

c = 512. As c decreases, the linear-scaling region be-comes narrower and the self-similarity becomes less ev-ident. Dependency of the correlation dimension on c is seen only for the global fractal, whereas the dimension is consistent across c values for the local fractals, as detailed in Supp. D.2. This difference in the behavior of local and global frac-tals suggests a fundamental difference between these two kinds. The local fractal does not depend on c, whereas the global fractal requires large c to appear. While the lo-cal fractal may stem from mixed word-frequency distribu-tions in topic models, as observed by Doxas et al. (2010) and mentioned above, the global fractal is due to long 5                       

> FIG. 2 Sequence of distributions ptunderlying the words in Don Quixote , as visualized for words “,” (comma) and “;” (semicolon). Each point represents one timestep. The green points represents timesteps at which pt(“,”) dominates and the Shannon entropy H(pt)<2.0, whereas the orange points correspond to high-entropy states with H(pt)>3.0. Self-similar patterns are observed in both the green and orange regions. 𝜈  = 6.5
> (a) (b) 𝜈  = 4(c) FIG. 3 Correlation integral curves as defined by Formula (2) and estimated with GPT2-xl with respect to (a) the maximum-probability threshold ηin Formula (8), (b) the sequence length N, and (c) the context length cin Formula (9).

memory and was anticipated in the literature (Altmann 

et al. , 2012; Li, 1989; Tanaka-Ishii and Bunde, 2016). Al-though self-similarity and long memory have often been studied separately and were even conjectured as different aspects of a scale-invariant process (Abry et al. , 2003), they show interesting coordination for natural language. More results on a larger dataset are provided in Supp. F.2. To further investigate the properties of natural lan-guage, we conducted a larger-scale analysis of long texts, which were divided into two groups: books in multi-ple languages and English articles in multiple genres, as detailed in Supp. E. The first group included 144 single-author books from Project Gutenberg and Aozora Bunko, comprising 80 in English, 32 in Chinese, 16 in German, and 16 in Japanese. The second group included 342 long English texts from different sources. We ob-tained all the results in this large-scale analysis by ap-plying the dimension-reduction method given in Formula (7). Figures 4 (a) and (b) show the large-scale results on the books for the correlation dimension ˆ ν with respect to (a) different languages and (b) various model sizes. The former results (a) were produced using the GPT2 model of size xl (denoting “extra-large”), with ≈ 10 9

parameters. For the latter results (b), we tested models of different sizes from small (≈ 10 6 parameters) to 34B 

(3 .4 × 10 10 ). For the sizes up to xl , we used the GPT2 model; for 6B and 34B , we used the Yi model (Yi, 2024), which offers the SOTA capability in English among all 6

(a) (b) 

(c) 

(d) (e) FIG. 4 Correlation dimensions of (a) all books grouped by language, as estimated using GPT2-xl ; (b) English books as estimated using GPT with different model sizes (GPT2 from small to xl and the Yi model for 6b and 34b ); (c) English texts from various sources with the R2 scores (horizontal axis) of their linear fits to the correlation integral curves; (d) shuffled English books evaluated with GPT2-xl ; and (e) English books evaluated with weight-randomized GPT2-xl .7publicly available LLMs. For all tested model sizes, the average correlation dimension remains constant. Out-liers occur more frequently for the two Yi models ( 6B 

and 34B ), which was possibly due to those models’ use of a lower numerical precision (16-bit floating-point num-bers). Hence, for all languages, an average correlation dimen-sion of around ˆ ν = 6 .5 is observed: 6 .39 ± 0.40 for En-glish, 6 .81 ± 0.58 for Chinese, 7 .30 ± 0.41 for Japanese, and 5 .84 ±0.70 for German ( ± indicates the standard de-viation). These results suggest the possible existence of a common dimension for natural language, with a lower bound of 6.5 under our settings. Figure 4(c) shows the correlation dimension (vertical axis) for English texts in four genres: books, academic papers (Kershaw and Koeling, 2020), the Stanford En-cyclopedia of Philosophy (SEP) 2, and Wikipedia web-pages. For each text, the horizontal axis indicates the coefficient of determination, R2, for the correlation in-tegral curve’s linear fit. A larger R2 value (maximum 1) implies more significant self-similarity in a text. The right side of (c) shows the distribution of the dimension values grouped by genre. As seen in the figure, most texts have a correlation di-mension around 6, especially those estimated with high 

R2 scores. The SEP texts (blue) have the most concen-trated range of dimensions, at 6 .57 ± 0.32 with R2 > 0.99 for over 90% of the texts. In contrast, the academic papers (black) show the most scattered distribution of the correlation dimension. This is deemed natural, as the SEP texts have the highest quality, whereas the aca-demic papers include irregular notations such as chemical and mathematical formulas, which obscure a text’s self-similarity. The universal correlation dimension value, ν ≈ 6.5, can be understood through the lens of the “information dimension” (Farmer, 1982), which coincides with ν un-der ergodic conditions (Pesin, 1993). The information dimension reflects how information, or the log count of unique contexts, scales with the statistical manifold’s res-olution. Contexts are deemed the same if their pt values are indistinguishably close within a certain threshold. Essentially, doubling the resolution would reveal about 26.5 ≈ 90 times more distinct contexts that were previ-ously considered identical. Therefore, ν quantifies the average “redundancy” in the diversity of texts conveying similar messages. We also compared several theoretical random pro-cesses. As analyzed using a GPT2-xl model and shown in Figure 4(d), shuffled word sequences exhibited an av-erage correlation dimension of 13.0, indicating inherent self-similarity despite the shuffling. As seen in Figure  

> 2https://plato.stanford.edu/

4(e), randomization of the GPT2-xl model’s weights sig-nificantly increased the correlation dimensions to an av-erage of 80. This result suggests purely random outputs, unlike text shuffling, which retains some linguistic struc-tures, like a bag-of-words approach. Analyses of additional random processes, as detailed in Supp. G, showed that a uniform white-noise pro-cess on the statistical manifold S yielded a correlation dimension over 100. Symmetric Dirichlet distributions in high-entropy regions consistently produced dimensions above 10. Conversely, Barab´ asi-Albert (BA) networks (Barab´ asi and Albert, 1999), which are special cases of a Simon process, demonstrated a correlation dimension of 2.00 ± 0.003, and a fractal variant (Rak and Rak, 2020) produced 2 ∼ 3.5. In terms of complexity via the corre-lation dimension, this places natural language above BA networks but below white noise. In Supp. H, we further investigate the relationship between the statistical manifold and conventional Eu-clidean spaces with respect to the correlation dimension. For BA models, the dimension remains the same whether measured in a Euclidean space or the manifold, thus em-phasizing the comparability. However, language data reveals a different story: Euclidean metrics yield com-promised linearity in comparison to Fisher-Rao metrics, thus underscoring that the Fisher-Rao distance more ac-curately captures language’s inherent self-similarity. Recently, LLMs have also been developed for process-ing data beyond natural language, and one successful ex-ample is for acoustic waves compressed into discrete se-quences (Copet et al. , 2023). To demonstrate the applica-bility of our analysis, we used the GTZAN dataset (Tzane-takis and Cook, 2002), which comprises 1000 recorded music pieces categorized in 10 genres. Briefly, we ob-served clear self-similarity in the compressed music data. The correlation dimension was found to depend on the genre: classical music showed the smallest dimension at 5.44 ± 1.13, much smaller than the dimensions for metal music at 7 .27 ± 0.96 and rock music at 7 .42 ± 0.87. None of the music genres showed a correlation dimension as large as that of white noise, as mentioned previously, even though the analysis was based on recorded data. The details of this analysis are given in Supp. I. In closing, we recognize this study’s limitation of view-ing text as a dynamical system akin to the GPT model, which overlooks the potential of representing words as leaf nodes in a syntactic tree, as suggested by generative and context-free grammars (CFGs) (Chomsky, 2014). Although promising, that complex linguistic framework exceeds our current scope, and we expect to explore it in the future. 8

ACKNOWLEDGMENTS 

This work was supported by JST CREST Grant Num-ber JPMJCR2114 and JSPS KAKENHI Grant Number JP20K20492. 

REFERENCES 

Abry, Patrice, Patrick Flandrin, Murad S Taqqu, et al. (2003), “Self-similarity and long-range dependence through the wavelet lens,” Theory and applications of long-range de-pendence 1, 527–556. Altmann, Edouard G, Giampaolo Cristadoro, and Mirko D. Esposti (2012), “On the origin of long-range correlations in texts,” Proceedings of the National Academy of Sciences 

109 (29), 11582–11587. Altmann, Eduardo G, Janet B. Pierrehumbert, and Adil-son E. Motter (2009), “Beyond word frequency: Bursts, lulls, and scaling in the temporal distributions of words,” PLoS One 4 (e7678). Amari, Shun-ichi (2012), Differential-geometrical methods in statistics , Vol. 28 (Springer Science & Business Media). Ausloos, Marcel (2012), “Measuring complexity with mul-tifractals in texts. translation effects,” Chaos, Solitons & Fractals 45 (11), 1349–1357. Balogh, Zolt´ an, and Annina Iseli (2016), “Dimensions of pro-jections of sets on riemannian surfaces of constant curva-ture,” Proceedings of the American Mathematical Society 

144 (7), 2939–2951. Barab´ asi, Albert-L´ aszl´ o, and R´ eka Albert (1999), “Emergence of scaling in random networks,” science 286 (5439), 509– 512. Blei, David M, Andrew Y Ng, and Michael I Jordan (2003), “Latent dirichlet allocation,” Journal of machine Learning research 3 (Jan), 993–1022. Brown, Tom, Benjamin Mann, Nick Ryder, et al. (2020), “Language models are few-shot learners,” Advances in neu-ral information processing systems 33 , 1877–1901. Chomsky, Noam (2014), Aspects of the Theory of Syntax , 11 (MIT press). Copet, Jade, Felix Kreuk, Itai Gat, et al. (2023), “Simple and controllable music generation,” in Thirty-seventh Confer-ence on Neural Information Processing Systems .Doxas, Isidoros, Simon Dennis, and William L Oliver (2010), “The dimensionality of discourse,” Proceedings of the Na-tional Academy of Sciences 107 (11), 4866–4871. Falconer, Kenneth (2004), Fractal geometry: mathematical foundations and applications (John Wiley & Sons). Farmer, J Doyne (1982), “Information dimension and the probabilistic structure of chaos,” Zeitschrift f¨ ur Natur-forschung A 37 (11), 1304–1326. Grassberger, Peter, and Itamar Procaccia (1983), “Charac-terization of strange attractors,” Physical review letters 

50 (5), 346. Kershaw, Daniel James, and R. Koeling (2020), “Elsevier oa cc-by corpus,” ArXiv abs/2008.00774 .Kobayashi, Tatsuru, and Kumiko Tanaka-Ishii (2018), “Tay-lor’s law for human linguistic sequences,” Proceedings of the 56th Annual Meeting of the Association for Computa-tional Lingusitics , 1138–1148. Lacasa, Lucas, and Jes´ us G´ omez-Gardenes (2013), “Correla-tion dimension of complex networks,” Physical review let-ters 110 (16), 168703. Li, Wentian (1989), “Mutual information functions of natural language texts,” (Citeseer). Marstrand, John M (1954), “Some fundamental geometrical properties of plane sets of fractional dimensions,” Proceed-ings of the London Mathematical Society 3 (1), 257–302. OpenAI, (2023), “Gpt-4 technical report,” arXiv:2303.08774 [cs.CL]. Osborne, A Ro, and A Provenzale (1989), “Finite correlation dimension for stochastic systems with power-law spectra,” Physica D: Nonlinear Phenomena 35 (3), 357–381. Peitgen, Heinz-Otto, Hartmut J¨ urgens, Dietmar Saupe, and Mitchell J Feigenbaum (1992), Chaos and fractals: new frontiers of science , Vol. 7 (Springer). Pesin, Ya B (1993), “On rigorous mathematical definitions of correlation dimension and generalized spectrum for dimen-sions,” Journal of statistical physics 71 , 529–547. Radford, Alec, Jeffrey Wu, Rewon Child, et al. (2019), “Lan-guage models are unsupervised multitask learners,” Ope-nAI blog 1 (8), 9. Rak, Rafal, and Ewa Rak (2020), “The fractional preferential attachment scale-free network model,” Entropy 22 (5), 509. Rao, C Radhakrishna (1992), “Information and the accuracy attainable in the estimation of statistical parameters,” in 

Breakthroughs in Statistics: Foundations and basic theory 

(Springer) pp. 235–247. Russell, David A, James D Hanson, and Edward Ott (1980), “Dimension of strange attractors,” Physical Review Letters 

45 (14), 1175. Simon, Herbert A (1955), “On a class of skew distribution functions,” Biometrika 42 (3/4), 425–440. Tanaka-Ishii, Kumiko, and Armin Bunde (2016), “Long-range memory in literary texts: On the universal clustering of the rare words,” PLoS One 11 (11), e0164658. Tanaka-Ishii, Kumiko, and Tatsuru Kobayashi (2018), “Tay-lor’s law for linguistic sequences and random walk models,” Journal of Physics Communications 2 (11), 089401. Touvron, Hugo, Louis Martin, Kevin Stone, et al. (2023), “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288. Tzanetakis, George, and Perry Cook (2002), “Musical genre classification of audio signals,” IEEE Transactions on speech and audio processing 10 (5), 293–302. Yi, (2024), “The yi model,” https://huggingface.co/ 01-ai/Yi-34B , visited in January 2024. 9

Appendix A: Properties of the Mapping ϕ : xt 7 → pt

1. Formulation 

In the main text, we consider two sequences of probability distributions, i.e., {xt} and {pt}, which are related by the mapping ϕ : xt 7 → pt in 5. We explain the formulation of ϕ here. Recall that xt denotes the language dynamical system’s state at timestep t and is defined as a distribution over the set Γ of all sequences of words, as in 3. Then, pt is defined over the vocabulary V and characterizes the probability of a word w to occur as the next word following a given context. The probability pt(w) for any w ∈ V is defined as the probability that an arbitrary closed text has w as its first word, thus giving the following formulation of ϕ:

pt(w) = ϕ(xt)( w) := X

> a≥t∈Γ
> at=w

xt(a≥t) ∀w ∈ V. (A1) 

xt and pt are defined over different sets but are essentially consistent with respect to the same probability measure 

μ on a probability space (Γ , F, μ ), where F = {Λ : Λ ⊂ Γ} denotes the power set of Γ. Here, μ : F → [0 , 1] is defined as follows: 

μ(Λ) = X

> a≥t∈Λ

xt(a≥t), ∀Λ ⊂ Γ. (A2) Hence, xt are pt are both consistent with respect to μ:

xt(a≥t) = μ({a≥t}) ∀a≥t ∈ Γ, (A3) 

pt(w) = μ

a

> a≥t∈Γ
> at=w

{a≥t}

 ∀w ∈ V. (A4) 

2. Linearity 

The mapping ϕ is linear with respect to the mixture of probability distributions within {xt}. That is, for any two distributions xt, x s, and any mixture weight α ∈ [0 , 1], ϕ satisfies the following: 

ϕ