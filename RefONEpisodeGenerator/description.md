이 프로젝트는 HM3D 기반의 MultiON 벤치마크의 에피소드를 생성하는 프로젝트로, goat-bench 처럼 벤치마크를 돌리려는게 아니라, 벤치마크에 사용될 에피소드를 만드는 거야.

~/PycharmProjects/goat-bench/README_explanation.md 를 보면 goat-bench 를 분석해놓은 문서가 있고, 여기에 내가 만들 벤치마크 (goat-bench 프로젝트를 복제해 수정할 예정) 가 받아들일 episode json 구조가 담겨있어.

그리고 이걸 바탕으로 ~/PycharmProjects/hm3d-scene-loader 폴더에 scene 의 정보를 가져오는 클래스를 만들어놨어. 그래서 scene id 와 cfg 만 있으면 scene 을 불러와서 각종 데이터를 가져올 수 있어. 이 프로젝트의 기능도 그대로 가져올거야.

그럼 이 프로젝트에서는 무엇을 하느냐, 라고 하면 에피소드 생성을 위한 파이프라인을 만들거야. 큰 틀에서는 MultiON 이라서, 한 에피소드에 subgoal 이 여러개 들어가게 되고, 이 subgoal 에는 종류가 있어서 구조화를 통한 생성이 가능해. (이미 pushdown automata 를 생성해 놓음)

하지만 현재 pushdown automata 를 완전히 그대로 구현하면 확장성이 떨어지므로, 새로운 종류의 subgoal 이 들어가도 작동하도록 oop 로 구현하려고 해. (각 subgoal 의 종류를 하나의 클래스로?)

내가 만드는 벤치마크의 핵심은 단순히 복수개 (N개) 의 object 를 N 개의 instruction 으로 지시하여 찾는게 아니라, 이전에 줬던 instruction 을 바탕으로 새로운 instruction 이 들어왔을 때 이전 instruction 정보를 참조하여 goal 을 추론해야 하는 식의 instruction 이라서, instruction 끼리 일종의 짝이 만들어져.

자세한 사항은 ./episode_generator_spec.md 를 참고해줘. 하지만 이 문서에 있는 instruction 종류로 끝나지 않고, 더 추가될 수 있으니 확장이 가능하게 만들어야 해.

추가로 만드려는 instruction 은 goal absent 형태로, 이전에 alias binding 을 하지 않았는데 이상한 alias 를 주면서 찾으라고 하거나, 아예 scene 에 없는 물체를 찾으라고 하거나, 아니면 두번째 subgoal 을 위한 instruction 인데 "세번째로 찾았던거 찾아줘" 라고 명령을 하는 등 말이 안되는 instruction 을 줄거야. 이런 경우 에이전트는 아예 멈추게 할거라서 그 뒤로는 subgoal 을 더 안만들어야 해. 이런 instruction 들도 가능하면 만들어줘.

grammer 가 담긴 문서를 봤으니 알겠지만 instruction 을 바로 만들어내는게 아니라, [plain_goal, AB_pre, plain_goal, AB_post] 와 같이 instruction style 이 담긴 리스트를 생성한 후에, 그 리스트 대로 instruction 을 끼워넣을거라서, 우선은 그 instruction style list 를 만들어내는 알고리즘이 필요해. 그리고 instruction style 을 정의하는 클래스에 실제 instruction 이 정의되면 좋겠지. AB_post 는 AB_pre 가 있어야만 한다와 같은 제약조건도 클래스에 정의되고, 만들어진 instruction style list 를 검증하는 절차도 이루어지면 좋을 것 같아.

리스트를 만드는 과정은 위와 같은데, 결국 이 리스트도 여러개 만들어야 하잖아? 그리고 여러 instruction style 들이 고르게 들어가면 좋겠고 instruction list 의 길이도 다양하고 고르게 분포되면 좋겠어서 좀 사이클을 돌면서 비율을 맞추는 작업을 하면 좋을 것 같아. 내가 config 로 주는건 각 instruction style length 의 비율과 각 scene 당 episode 의 개수로, grammer 상 정확히 개수를 맞춰서 list 를 만드는게 어려울거라서, 그냥 무작위로 많이 만들어버리고, 통계 내고, 버릴거 버리고 아니면 더 만들어서 조건에 맞는거 더 추가하고, instruction style 들의 개수도 통계 지속적으로 내면서 확인하면 될 것 같아.

다 만들고 나면 전체적인 통계를 보여주면 될 것 같아.

이 프로젝트의 명령어는 두개로 나뉘는데,
1. instruction style list generating
2. instruction style list to real hm3d episode json

이렇게 두개로 나뉘어져서, 1번에서는 instruction style list 들만 그 사이클 돌면서 생성해내고, 생성한 리스트들을 어딘가 폴더에 저장, 폴더에 저장 시에는 해당 폴더에 statistic 도 같이 저장해야 해. 추가로 몇개의 scene 을 위한 episode 를 만들건지도 인자로 받아야겠네. 즉 한 scene 에 대해서도 여러 에피소드가 만들어질 수 있고, 각 에피소드에는 여러 subgoal 이 있는 방식이야.

두번째는 첫번째에서 생성한 instruction style list 가 담긴 폴더명을 인자로 넘기면 해당 폴더에 생성된 instruction style list 를 실제 hm3d episode json 으로 만들어내야 하는데, 이 때에는 각 instruction 에 매칭될 object 들이 필요하므로 hm3d scene loader 가 필요할거야. 해당 episode 에 필요한 goal 들을 가져와서 scene 내의 랜덤한 goal 들에 매칭시키면 돼.

아 그리고 ./episode_generator_spec.md 에 보면 Generation Algorithm 섹션이 있을텐데, 이는 무시해도 되니 아까 말한대로 oop 형태로 제작해줘.

설명이 부족하거나 궁금한 점 있으면 꼭 물어보고.