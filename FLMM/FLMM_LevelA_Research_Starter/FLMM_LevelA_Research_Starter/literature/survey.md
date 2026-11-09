# Focused literature matrix (v0)

| Work | Year/venue | Core task | Why read | Baseline role | Source |
|---|---|---|---|---|---|
| F-LMM | CVPR 2025 | Frozen LMM image grounding, RES/PNG | Main architecture and original benchmark | **Primary** | https://github.com/wusize/F-LMM |
| SAM | ICCV 2023 | Promptable image segmentation | F-LMM's SAM-based refinement | Component context | https://github.com/facebookresearch/segment-anything |
| SAM 2 | 2024 | Image/video promptable segmentation | Future video propagation, not text grounding alone | Later extension | https://github.com/facebookresearch/sam2 |
| Grounded SAM 2 | 2024 pipeline | Text grounding + tracking | Practical alternative for video masks | Contingency only | https://github.com/IDEA-Research/Grounded-SAM-2 |
| VideoGLaMM | CVPR 2025 | Pixel-level video visual grounding | What trained video architecture can do | Related work | https://openaccess.thecvf.com/content/CVPR2025/html/Munasinghe_VideoGLaMM__A_Large_Multimodal_Model_for_Pixel-Level_Visual_Grounding_CVPR_2025_paper.html |
| DecAF | ICLR 2026 | Training-free video reasoning segmentation via attention fusion | Important overlap with 'attention -> video masks' ideas | Related work, not same task | https://proceedings.iclr.cc/paper_files/paper/2026/hash/2fc6b8a3fc23108f184daa4759024c25-Abstract-Conference.html |

**Scientific caution:** SAM and SAM2 are different models; a method with video propagation is not automatically a referring-expression video-grounding model. Distinguish image masks, video masks and temporal intervals.

## Reading assignments (one-page notes per paper)
Paper question | setup | metrics/splits | main contribution | code/checkpoint availability | assumptions | weakness that can be independently tested | cost to reproduce.
