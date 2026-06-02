

VILNIUS GEDIMINAS TECHNICAL UNIVERSITY  
Faculty Of Fundamental Sciences  
Department Of Information Technologies

Navid Shad Gorabsari

*BESIMOKANČIOJO LINGVISTINIO ŽINIŲ GRAFO MODELIAVIMAS IR VIZUALIZAVIMAS, TAIKANT VEKTORIZAVIMĄ IR GRAFŲ INTEGRAVIMĄ*

*MODELING AND VISUALIZATION OF A LEARNER’S LINGUISTIC KNOWLEDGE GRAPH USING EMBEDDING AND GRAPH INTEGRATION*

Master’s Thesis Work 1 

*Master’s in Engineering of Artificial Intelligence*  
*Informatics*

VILNIUS GEDIMINAS TECHNICAL UNIVERSITY  
Faculty Of Fundamental Sciences  
Department Of Information Technologies

Navid Shad Gorabsari

*BESIMOKANČIOJO LINGVISTINIO ŽINIŲ GRAFO MODELIAVIMAS IR VIZUALIZAVIMAS, TAIKANT VEKTORIZAVIMĄ IR GRAFŲ INTEGRAVIMĄ*

*MODELING AND VISUALIZATION OF A LEARNER’S LINGUISTIC KNOWLEDGE GRAPH USING EMBEDDING AND GRAPH INTEGRATION*

Master’s Thesis Work 1 

*Master’s in Engineering of Artificial Intelligence*  
*Informatics*

| Supervisor | Professor Simona Ramanauskaitė |  |
| ----- | :---: | ----- |
|  | (Academic degree/title, name, surname) |  |
| **Consultant** |  |  |
|  | (Academic degree/title, name, surname) |  |

**Content**

**[1\. Introduction	4](#1.-introduction)**

[1.1 Background and Motivation	4](#1.1-background-and-motivation)

[1.2 Scientific Problem	4](#1.2-scientific-problem)

[1.3 Research Aim, Object and Objectives	6](#1.3-research-aim,-object-and-objectives)

[1.4 Practical Value and Implementation Benefits	6](#1.4-practical-value-and-implementation-benefits)

[1.5 Research Questions	7](#1.5-research-questions)

[1.6 Significance of the Study	8](#1.6-significance-of-the-study)

[1.7 Relevance and Novelty Justification	8](#1.7-relevance-and-novelty-justification)

[1.8 Research Scope and Limitations	8](#1.8-research-scope-and-limitations)

[**2\. Literature Review	9**](#2.-literature-review)

[2.1 Overview of Knowledge Graphs in AI and Education	9](#2.1-overview-of-knowledge-graphs-in-ai-and-education)

[2.2 Evolution of Knowledge Tracing: From Sequences to Graphs	9](#2.2-evolution-of-knowledge-tracing:-from-sequences-to-graphs)

[2.3 Constructing Educational Graphs: Heterogeneous Approaches	10](#2.3-constructing-educational-graphs:-heterogeneous-approaches)

[2.4 Semantic Representation: BERT, K-BERT, and LLMs	11](#2.4-semantic-representation:-bert,-k-bert,-and-llms)

[2.5 Using Graphs to Simulate Thinking (GNNs)	12](#2.5-using-graphs-to-simulate-thinking-\(gnns\))

[2.6 Modeling Dynamics: Time and Forgetting	13](#2.6-modeling-dynamics:-time-and-forgetting)

[2.7 Personalization and Recommendation Systems	13](#2.7-personalization-and-recommendation-systems)

[2.12 Summary	17](#2.12-summary)

[**3\. Methodology – Conceptual Framework for Neural–Graph Integration	18**](#3.-methodology-–-conceptual-framework-for-neural–graph-integration)

[3.1 Research Methods and Justification	18](#3.1-research-methods-and-justification)

[3.2 Conceptual Nature and Scientific Basis	20](#3.2-conceptual-nature-and-scientific-basis)

[3.3 System Overview	21](#3.3-system-overview)

[3.4 Semantic Embedding and Knowledge Injection	22](#3.4-semantic-embedding-and-knowledge-injection)

[3.5 Graph Neural Reasoning Layer	22](#3.5-graph-neural-reasoning-layer)

[3.6 Learner Knowledge State Update	23](#3.6-learner-knowledge-state-update)

[3.7 Gap Detection and Adaptive Path Recommendation	24](#3.7-gap-detection-and-adaptive-path-recommendation)

[3.8 Personalization via Knowledge Graph Tuning (KGT)	24](#3.8-personalization-via-knowledge-graph-tuning-\(kgt\))

[3.9 Theoretical Feasibility and Expected Implementation Path	25](#3.9-theoretical-feasibility-and-expected-implementation-path)

[References	26](#references)

# **1\. Introduction** {#1.-introduction}

## **1.1 Background and Motivation** {#1.1-background-and-motivation}

Nowadays, artificial intelligence and digital education are growing fast. This gives us a chance to understand how humans learn languages by using neural networks and knowledge graphs. Traditional methods for learning languages usually have a fixed structure. They use set grammar rules and give generic feedback. But in the real world, learning is not like that. It is personal and depends on the context. Learners get knowledge by reading, writing, and speaking. Every time they do these things, their understanding changes a little bit. We can show these changes using graph-based computer models.

Knowledge Graphs (KGs) are good tools to show complex knowledge. They work like a network of connected ideas. When we combine them with neural networks, like BERT models, we get a system that can understand meaning and learn from the user. This combination helps the AI find gaps in what the student knows and suggest the best way to learn.

This research proposes an **AI-driven model** to build a "linguistic knowledge graph" for each learner. The goal is to show how a student learns, forgets, and organizes English words and rules over time. By using neural networks to find similar meanings and graph networks to connect them, the system can simulate how a student understands English.

## **1.2 Scientific Problem** {#1.2-scientific-problem}

Technology in education has improved, but most learning systems still have important limits. They usually look at scores or specific mistakes without seeing how different parts of language connect to each other. This creates a gap in understanding the real structure of what students know.

The main problem is representing knowledge dynamically. We need to show how a learner's skills change over time. For example, if a student learns verb tenses well, it helps them understand adverbs and sentence structures too. Normal systems cannot see these connections.

Building effective knowledge graphs for language learning faces several critical challenges. Recent systematic reviews identify major problems:

1. **Lack of standardization**: Educational knowledge graphs do not have a common structure like healthcare or finance domains. This makes sharing knowledge graphs across different platforms very difficult.

2. **Sparse and incomplete data**: Many knowledge graphs in education contain incomplete information. Some concepts are well-represented, but others have limited or no coverage.

3. **Limited interoperability**: Educational institutions build their own knowledge graphs that work alone. They do not connect with other systems.

4. **Scalability challenges**: As knowledge graphs grow, they become difficult to manage, search, and update. Educational institutions create huge amounts of data every day.

5. **Semantic heterogeneity**: Educational knowledge graphs work with data from many different sources. Student performance might be called "learner achievement" or "academic progress" in different places.

6. **Real-time updates remain technically demanding**: Educational knowledge is not static. It changes with new research and teaching trends. For language learning, students learn and forget words at different rates. The system needs to track these changes continuously.

7. **Implicit relationships are hard to capture**: Implicit relationships cannot be directly obtained through existing data description. They need relationship calculation, data fusion, and prediction models. These invisible connections affect the accuracy of decisions.

These problems create the core scientific challenge: How can a system based on neural networks build and update a personal knowledge graph that shows a learner's changing English skills and finds what they are missing, while handling sparse data, lack of standardization, scalability issues, semantic differences, real-time updates, and implicit relationships?

Current solutions are not enough. Some systems use only rules (good for accuracy, bad for flexibility), only deep learning (good for context, bad for logic), or only graphs (good for reasoning, needs lots of data). The literature shows hybrid approaches work best. However, most hybrid studies focus on search engines or shopping recommendations. There is little research on using knowledge graphs with BERT and graph neural networks specifically for English grammar tutoring.

This research addresses these gaps by proposing an AI-driven model that builds a linguistic knowledge graph for each learner. The system will show how a student learns, forgets, and organizes English words and rules over time.  
​

## **1.3 Research Aim, Object and Objectives** {#1.3-research-aim,-object-and-objectives}

**Research aim**: To design and compare technical options for a system that can build and update a personal linguistic knowledge graph for English learners, so the system can represent changing knowledge and detect learning gaps over time.

**Research object**: The object of this research is the AI-based modeling of a learner’s linguistic knowledge graph, including how linguistic concepts are represented, connected, updated, and personalized using neural and graph-based methods.

**Research objectives:**

1. To review and compare main educational KG construction approaches (rule-based, statistical, neural, and hybrid) and decide which ones fit language learning best.

2. To define what the nodes and edges should represent in a learner’s linguistic KG (e.g., grammar rules, vocabulary concepts, prerequisites, and semantic links).

3. To compare semantic representation options for language input (e.g., BERT vs. knowledge-injected embeddings such as K-BERT) for mapping learner text/errors to graph concepts.

4. To examine graph-learning options (e.g., GNN-based propagation) for updating relationships between concepts and supporting gap detection.

5. To explore update strategies for dynamic learning (including forgetting and implicit relationships from learner behavior) and how these can be reflected in the graph.

6. To define a lightweight personalization approach (e.g., updating the learner’s graph without retraining the whole model) as a practical technical option.

## **1.4 Practical Value and Implementation Benefits** {#1.4-practical-value-and-implementation-benefits}

This research is not just theory. It has practical value according to the Oslo Manual’s definition of innovation. It creates a new or improved process that is useful for users. The model solves real problems in online education and tutoring systems.

### **1.4.1 Value for Online Learning Platforms**

The main benefit is for platforms like **Subturtle** that offer adaptive English learning. Current platforms have three big problems:

1. They cannot show complex connections between language concepts.  
2. They do not adapt fast enough to the learner.  
3. They do not explain why they recommend certain content.

Our proposed framework solves these issues. It treats language concepts as nodes in a network. This turns learning from a straight line into a dynamic map. It helps learners understand *why* they need to learn one thing before another. Also, developers can use **Knowledge Graph Tuning** (Sun et al., 2024). This allows the system to update personal graphs without retraining the whole model. This saves computing power and money.

### **1.4.2 Value for Educators and Schools**

For teachers, this graph gives a clear picture of what students understand. Instead of just looking at test scores, teachers can see a map of mastered concepts. They can see which topics the students understand and which ones are difficult. This helps teachers plan better lessons.

Also, the model explains its decisions. This is important for AI in education. Teachers can see the logic behind the system's recommendations. This builds trust between the AI and the teacher.

### **1.4.3 Value as an Innovation**

This system is an innovation in two ways:

* **Product innovation:** It is a new tool for learning English.  
* **Process innovation:** It is a new way to model learners. To our knowledge, no other commercial platform uses K-BERT and GNNs specifically for personal English learning. It combines techniques that are usually separate.

### **1.4.4 Implementation Roadmap**

We can implement this in three phases:

* **Phase 1:** Use the graph to visualize the curriculum for teachers.  
* **Phase 2:** Add the graph neural network to make personalized recommendations.  
* **Phase 3:** Use Knowledge Graph Tuning for real-time updates.

## 

## **1.5 Research Questions** {#1.5-research-questions}

To achieve the research aim of designing and comparing technical options for a personal linguistic knowledge graph, this study answers the following questions:

1. Which knowledge graph construction approach (rule-based, statistical, neural, or hybrid) is most suitable for modeling the complex interdependence of English grammar and vocabulary rules?​

2. How do semantic representation models differ in their ability to map learner errors to graph nodes, and does injecting external knowledge (e.g., K-BERT) provide a significant advantage over standard embeddings (e.g., BERT)?​

3. Which graph neural network (GNN) mechanism is most effective for propagating learner mastery scores across related concepts to detect implicit knowledge gaps?​

4. How can dynamic learning behaviors (such as the forgetting curve and skill decay) be technically integrated into the graph update rules to ensure the model reflects the learner’s real-time state?​

5. What is the most computationally efficient technical strategy for personalization—retraining the entire model per user, or using lightweight methods like Knowledge Graph Tuning (KGT)?​

## 

## **1.6 Significance of the Study** {#1.6-significance-of-the-study}

This research is important for theory and practice. Theoretically, it adds to the knowledge of **neural-symbolic integration**. This means combining the power of deep learning with the logic of graphs. Pedagogically, it gives a plan for building intelligent tutoring systems.

Also, the model supports **Explainable AI (XAI)**. It ensures that recommendations are clear. The graph visually shows why an exercise is recommended. This makes the system more transparent for the learner.

## 

## **1.7 Relevance and Novelty Justification** {#1.7-relevance-and-novelty-justification}

**Relevance**

The relevance of this study lies in the urgent need to improve online education. Current platforms often use a "one-size-fits-all" approach that treats every student the same. This research addresses this problem by proposing a Personal Knowledge Graph that tracks exactly what a student knows and forgets. This is significant because it shifts the technology from a static digital textbook to an adaptive intelligent tutor that improves learning efficiency.​

**Novelty**

This research introduces three specific innovations to the field of educational AI:

1. Domain Focus: While most recent studies on Knowledge Graphs focus on logical subjects like Math or Programming, this research specifically targets the complex, context-dependent nature of English language learning.​

2. Hybrid Technology: This model is one of the first to theoretically combine K-BERT (for understanding the meaning of text) with Graph Neural Networks (for connecting grammar rules). This solves the issue where AI understands structure but misses the context.​

3. Real-Time Adaptation: It proposes using "Knowledge Graph Tuning" for education. This allows the system to update a learner's personal graph instantly without the high cost of retraining the entire model, making personalized AI affordable and scalable.

## **1.8 Research Scope and Limitations** {#1.8-research-scope-and-limitations}

This study is **theoretical**. We do not collect real data or build the full software because of time limits. However, the design is based on proven principles from other research. It is a foundation for future work on the Subturtle platform.

The scope is limited to **Intermediate English** (grammar, vocabulary, and semantics). It might work for other languages later, but right now we focus only on English.

# **2\. Literature Review** {#2.-literature-review}

## **2.1 Overview of Knowledge Graphs in AI and Education**  {#2.1-overview-of-knowledge-graphs-in-ai-and-education}

To understand this research, we first need to define two main concepts: Knowledge Graphs (KGs) and Knowledge Tracing (KT). In the field of Artificial Intelligence, a Knowledge Graph is a way of organizing information. Instead of storing data in simple lists or tables, a graph looks like a web. It consists of "nodes" (which represent concepts, exercises, or students) and "edges" (which represent the relationships between them). For example, a graph might connect the grammar rule "Past Tense" to the word "Yesterday" to show they are related.

This structure is very important for Knowledge Tracing (KT). KT is a core technology in intelligent education. Its main goal is to track how a student’s knowledge changes over time based on their history of answering questions. Traditional systems just look at test scores, but KT tries to predict if a student will answer the *next* question correctly. When we combine these two technologies, we get a powerful tool. A Knowledge Graph provides the "map" of the subject, and Knowledge Tracing tracks the student's location on that map. This allows the system to see connections. If a student understands "Present Simple," the graph can predict that they might be ready for "Present Continuous" because the nodes are connected.

## **2.2 Evolution of Knowledge Tracing: From Sequences to Graphs** {#2.2-evolution-of-knowledge-tracing:-from-sequences-to-graphs}

The technology for tracking student learning has changed a lot in recent years. Based on the literature, we can divide this evolution into three main stages: traditional methods, deep sequential learning, and the modern graph-based approach.

### **2.2.1 Deep Sequential Learning (DKT)**

With the rise of Deep Learning, researchers created Deep Knowledge Tracing (DKT). This method uses Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) models. Instead of looking at concepts one by one, DKT looks at the student's whole history as a timeline. It is very good at finding patterns over time, such as how a student improves after practice. However, DKT has a major weakness. It treats questions and concepts as simple ID numbers without understanding how they are related. For example, it does not know that "car" and "bus" are similar words; it just sees them as "Item 1" and "Item 2".

### **2.2.2 Graph-Based Learning (GKT)**

The most recent stage is Graph-based Knowledge Tracing (GKT). This approach solves the problems of the previous models by using Graph Neural Networks (GNNs). In this model, knowledge concepts are nodes in a graph, and the system learns how they influence each other. Recent studies, such as the SCDI model (2025), use "heterogeneous graphs." This means the graph includes different types of nodes—students, exercises, and concepts—all connected in one network. Other models, like DyGKT (2024), introduce "dynamic graphs" that change continuously as the student learns .

## **2.3 Constructing Educational Graphs: Heterogeneous Approaches** {#2.3-constructing-educational-graphs:-heterogeneous-approaches}

In the past, educational graphs were simple. They usually connected just one type of thing, like "Concept A" is related to "Concept B." However, recent research shows that this is not enough. To truly understand a student, we need Heterogeneous and Multilayer Graphs.

### **2.3.1 What is a Heterogeneous Graph?**

A heterogeneous graph is a network that contains different types of nodes and edges. Instead of just mapping grammar rules, these modern graphs connect three main elements: Students, Exercises, and Concepts. For example, the SCDI model (2025) builds a graph where a "Student" node connects to an "Exercise" node (showing they attempted it), and the "Exercise" node connects to a "Concept" node (showing what the question tests).

### **2.3.2 Capturing Hidden Relationships**

Another challenge is that some relationships in education are not obvious. This is called "implicit" information. A study on Multilayer Knowledge Graphs (2025) argues that we cannot just look at the official curriculum. We also need to look at student behavior. If many students who fail "Question A" also fail "Question B," there is a hidden connection between them, even if the textbook does not say so. New models like DGEKT (2024) take this further by using "Hypergraphs," where a single connection can group many nodes together at once, which is very useful for language sentences that use multiple grammar rules simultaneously . Other researchers have also proposed "multisource hierarchical" networks to better organize these complex layers .

## **2.4 Semantic Representation: BERT, K-BERT, and LLMs** {#2.4-semantic-representation:-bert,-k-bert,-and-llms}

Once we have the structure of the graph, the computer needs to understand the meaning of the text inside the nodes. This is where Semantic Representation comes in.

### **2.4.1 Understanding Context with BERT**

Traditional methods used simple ID numbers to represent words. This was bad for language learning because the computer did not know that "speak" and "talk" have similar meanings. To fix this, researchers now use Pre-trained Language Models (PLMs) like BERT. BERT is a powerful AI model that reads text to understand context. However, BERT guesses patterns based on statistics, but it does not technically "know" facts (e.g., it might not know Paris is in France unless told).

### **2.4.2 Injecting Knowledge into AI**

To solve this, researchers are combining BERT with Knowledge Graphs. This is often called "Knowledge Injection." For example, K-BERT uses a knowledge graph to add extra facts to the sentence before the AI reads it. A recent model called ShallowBKGC (2025) uses BERT to read the text description of a concept and then uses that understanding to fix missing links in the Knowledge Graph. Other studies focus on "relation-guided aggregation," which helps the AI understand *how* two words are related, not just *that* they are related . This hybrid approach—using the logic of graphs and the language skills of BERT—is essential for building a good English tutor.

## **2.5 Using Graphs to Simulate Thinking (GNNs)** {#2.5-using-graphs-to-simulate-thinking-(gnns)}

Building a graph is only the first step. To make the system "smart," we need a way to process the information inside it using Graph Neural Networks (GNNs).

### **2.5.1 The Message Passing Mechanism**

In a GNN, information does not stay in one place. If a student answers a question about "Past Tense" correctly, the GNN passes a "message" to all connected nodes, such as "Present Perfect." The system mathematically calculates that if the student knows the first topic, their probability of knowing the related topics increases .

### **2.5.2 Advanced GNN Architectures**

Recent research has developed specialized GNNs for education. A major breakthrough is the DGEKT (Dual Graph Ensemble) model (2024). It uses two graphs: a Concept Hypergraph (grouping exercises by topic) and a Directed Transition Graph (tracking the order of learning). Another important model is DPDG-GKT (2025). This paper highlights a problem with standard GNNs: they can be unstable when updating student knowledge. To fix this, the authors added a "gating mechanism" that acts like a valve to control information flow, ensuring one lucky guess does not trick the system.

## **2.6 Modeling Dynamics: Time and Forgetting** {#2.6-modeling-dynamics:-time-and-forgetting}

Learning is not static; it changes over time. A student might know a word today but forget it next week. Therefore, modern systems must move from static graphs to Dynamic Graph Learning.

### **2.6.1 Beyond Simple Forgetting Curves**

Older systems used a simple mathematical formula called the "Ebbinghaus Forgetting Curve." While useful, this is too simple for complex language learning. Newer research treats the learning process as a Continuous-Time Dynamic Graph (CTDG) . Other studies use "Temporal Inductive Path Neural Networks" to predict how relationships change over time.

### **2.6.2 Long-term vs. Short-term Knowledge**

A critical distinction in the literature is between long-term mastery and short-term memory. A study by Yu et al. (2025) proposes the L-SKSKT model, which explicitly separates these two states. It uses a short-term graph for recent quizzes and a long-term graph for stable knowledge accumulated over months. This allows the AI to distinguish between a student who is "cramming" and a student who truly understands the material.

## **2.7 Personalization and Recommendation Systems** {#2.7-personalization-and-recommendation-systems}

Building a Knowledge Graph gives us a map, but the student needs a guide. This is the job of the Recommendation System.

### **2.7.1 Finding the Best Learning Path**

The goal is to find the most efficient path through the graph. Recent research by Zhou and Wang (2025) uses Deep Reinforcement Learning (DRL) for this task. In this approach, the AI acts like a player in a game, trying different paths to maximize the student's test score. Other researchers have developed "Intelligent Guide Applications" that actively steer students away from materials that are too hard or too easy.

### 

### **2.7.2 Explainable Recommendations**

One big problem with older AI systems is that they are "black boxes." Knowledge Graphs solve this by providing Explainable Recommendations. A study on MOOCs (2025) shows that the system can explain its logic: "We recommend this video because it explains a concept you missed in the last quiz".

### **2.7.3 Recommending Resources**

Finally, the system must recommend actual learning materials. A 2025 study on Advanced Knowledge Graphs demonstrates how to link specific resources (like videos or articles) to the graph nodes, ensuring the content matches the student's exact difficulty level.

### **2.8 The Idea of a Personal Knowledge Graph**

Most research focuses on "Global Knowledge Graphs," which are huge maps of all information in the world (like Wikipedia). However, for education, we need something smaller and more specific: a **Personal Knowledge Graph (PKG)**.

According to Balog and Kenter (2019), a Personal Knowledge Graph is different because it is "user-centric." It does not just store facts; it stores the *user's relationship* to those facts. For an English learner, the graph is not just a dictionary. It is a record of which words the user knows, which ones they forget, and which ones they use often. This is a key concept for this thesis. Instead of one big graph for everyone, we propose a small, private graph for each student.

A major challenge with personal graphs is keeping them updated. If we have 1,000 students, we cannot retrain a massive AI model 1,000 times a day. Sun et al. (2024) proposed a solution called **"Knowledge Graph Tuning" (KGT)**. Instead of retraining the whole brain of the AI, they just update the connections in the small personal graph. This method is fast and efficient, making it possible to have a personalized AI tutor that adapts in real-time.

### **2.9 Connecting AI to Psychology**

The best AI systems do not just do math; they mimic how humans learn. Recent studies have started connecting computer science with educational psychology.

### **2.9.1 The Role of Feedback**

Feedback is essential for learning. A very new study by Nongkhai et al. (2026) tested different ways of giving feedback to students. They found that a "Hybrid GenAI-Adaptive" approach worked best. This means using a Knowledge Graph to find the student's mistake and then using Generative AI (like ChatGPT) to explain it in simple language. Students who used this system made fewer mistakes than those who used traditional methods.

### **2.9.2 Cognitive Assessment**

AI can also assess *how* a student writes, not just *what* they write. Zhang (2025) developed a system for "AI Writing Assessment." It uses a graph to check if the student's essay has a logical structure and good vocabulary. The study found that when the AI understands the "semantics" (meaning) of the text using a graph, it can give feedback that feels like a human teacher. This supports the idea that our model needs to understand both the structure of grammar and the psychology of feedback.

### **2.10 Comparing the Options**

To choose the best method for this thesis, we compared the different approaches found in the literature. The table below summarizes the technical options:

| Feature | Traditional Methods (DKT/RNN) | Static Knowledge Graphs | Dynamic/Hybrid Graphs (Our Focus) |
| :---- | :---- | :---- | :---- |
| **Structure** | Sequential (Time-based) | Relational (Concept-based) | **Both (Time \+ Concept)** |
| **Logic** | Finds patterns, but cannot reason. | Good reasoning, but ignores time. | **Reasoning \+ Forgetting Curve** |
| **Input Data** | Student scores only. | Curriculum structure only. | **Student behavior \+ Text semantics** |
| **Personalization** | Low (Same model for all). | Medium (Path changes). | **High (Unique graph per user)** |
| **Key Citations** | Piech (2015) | Dessì (2021) | **Yu (2025)** |

The literature clearly shows that the **Dynamic/Hybrid** approach is superior. It combines the "memory" of sequential models (handling forgetting) with the "logic" of graphs (understanding grammar rules).

### **2.11 What is Missing?**

Although there is a lot of research, there is a specific gap that this thesis will address.

1. **Lack of English-Specific Personal Graphs:** Most studies focus on Math or Programming (like Nongkhai ). There are very few systems that build a *personal* graph specifically for English grammar and vocabulary.

2. **Semantic \+ Structural Disconnect:** Some papers use BERT to understand text, and others use GNNs to recommend paths. However, very few combine **K-BERT (Knowledge Injection)** with **Personal Knowledge Graphs** to teach a language.

3. **Real-Time Updating:** While Sun proposed "Knowledge Graph Tuning," it has not been fully tested in an educational app for language learners.

This research aims to fill these gaps by designing a model that is personal, linguistic, and dynamic.

## **2.12 Summary** {#2.12-summary}

This literature review has systematically examined the evolution of educational AI from static, rule-based systems to dynamic, graph-based models. The analysis of 30 key sources leads to three critical conclusions that define the direction of this thesis. First, while Knowledge Tracing (KT) has evolved significantly, most current systems still treat students as generic users rather than maintaining a Personal Knowledge Graph (PKG) for each individual. Second, although Graph Neural Networks (GNNs) are powerful for finding hidden connections , they are rarely combined with Knowledge Injection (K-BERT) in the specific domain of English language learning. Third, the psychological aspect of learning—specifically the need for personalized, generative feedback—remains underutilized in pure graph models.​

Therefore, the literature confirms that a gap exists for a hybrid system that is simultaneously semantic (understanding text meaning), structural (connecting grammar rules), and dynamic (adapting to forgetting). This thesis will address this gap by proposing a "Linguistic Knowledge Graph" model. The following section, Methodology, will detail the conceptual framework for building this system, specifically focusing on how to integrate K-BERT embeddings with a dynamic GNN to simulate a human tutor's understanding of a learner's evolving mind.

# **3\. Methodology – Conceptual Framework for Neural–Graph Integration** {#3.-methodology-–-conceptual-framework-for-neural–graph-integration}

This section is based on only 6 literature review which has been done earlier.

## **3.1 Research Methods and Justification** {#3.1-research-methods-and-justification}

### **3.1.1 Why We Chose a Theoretical Approach**

This research uses a theoretical and conceptual methodology. According to the FMF Scientific Research Handbook, this type of research is designed to analyze theories and concepts to create new models. Specifically, this study combines "descriptive" research (describing a new system) with "causal-comparative" research (explaining how this new system causes better learning results).

We chose a theoretical approach for several practical and scientific reasons.

First, this is a foundational design phase.  
In engineering, before you can build a complex machine, you need a blueprint. You cannot simply start coding a massive AI system without first proving that the design makes sense. According to the handbook, theoretical research is "deductive." This means we start with general proven rules—like how Knowledge Graphs work (Abu-Salih, 2024\) or how Neural Networks process text (Liu et al., 2019)—and we apply them to a specific new problem: teaching English. By doing this, we can design a rigorous framework without needing the massive supercomputers or years of data collection that a full experiment would require.

Second, we can validate the system through logic.  
Even though we are not testing this on real students yet, we can still prove the system works by looking at its parts. The architecture is like a puzzle made of pieces that have already been tested:

We know K-BERT works for understanding meaning (proven by Liu).

We know Graph Neural Networks (GNNs) work for finding connections (proven by Kipf & Welling).

We know Reinforcement Learning works for personalization (proven by Zhou & Wang).

Since every individual part of the machine has been proven to work in peer-reviewed studies, we can logically argue that putting them together will also work. Section 4 of this thesis will provide a "simulation"—a walkthrough of how the system handles data—to demonstrate this logic in action.

### 

### **3.1.2 Being Transparent About Limitations**

Every research method has limits, and it is important to be honest about them. The main limitation of this study is that it is predictive, not demonstrative.

Because this is a theoretical paper, we have not built the software or tested it in a real classroom. We do not have log files from real students. This means that while we predict the system will be effective, we cannot point to a chart and say, "Look, test scores went up by 20%."

We are also making some "idealized assumptions."

Perfect Data: We assume the computer can read student essays without making mistakes. In the real world, students might use slang or make typos that confuse the AI.

Stable Learners: We assume that if a student forgets a word, it follows a standard mathematical curve. In reality, students get tired, bored, or distracted, which affects how they learn.

Scope: Currently, this design is only for Intermediate English. We don't know yet if it would work for a language with a totally different structure, like Chinese or Arabic.

How we address these limits:  
To make sure this research is still scientifically valid, we have taken three specific steps:

Reliance on Proven Science: We never guess. Every single part of the design is based on a technology that already exists and has been verified by other scientists.

The "Walkthrough" Simulation: In Section 4, we don't just talk about the theory; we walk through a specific scenario step-by-step. This acts as a "Proof of Concept."

Clear Documentation: We list every assumption we make. This is useful for future researchers. If someone wants to build this system later, they will know exactly which parts need to be tested.

### **3.1.3 Matching the Method to the Objectives**

Our choice of methodology aligns perfectly with the four goals we set at the beginning of this thesis:

* Objective 1: We wanted to analyze the theory behind Knowledge Graphs. We achieved this by conducting a deep literature review and synthesizing existing theories.

* Objective 2: We wanted to propose a new design. We achieved this by describing the conceptual framework in this chapter, separate from the actual coding.

* Objective 3: We needed a way to find "gaps" in knowledge. We achieved this by adapting mathematical formulas from graph theory to measure "distance" between concepts.

* Objective 4: We wanted to simulate a learning path. We achieved this by creating the detailed scenario in Section 4\.

### 

## **3.2 Conceptual Nature and Scientific Basis** {#3.2-conceptual-nature-and-scientific-basis}

This methodology presents a **theoretical yet technically grounded framework** for modeling linguistic knowledge acquisition through neural–graph integration. Although the implementation remains conceptual, each module is inspired by peer-reviewed and validated research. The design is meant to bridge theory and practice — outlining a system architecture that can be realized in future empirical work, particularly within the Subturtle platform.

The approach does not propose speculative mechanisms but rather **recombines existing, proven concepts** into a new configuration tailored for personalized language learning. The methodology thus serves as a *blueprint for implementation* rather than an experimental prototype.

**Scientific foundations of each component:**

| Component | Scientific Source | Validation Status |
| ----- | ----- | ----- |
| Semantic Embedding (BERT/K-BERT) | Liu et al. (2019) – *arXiv:1909.07606* | Proven NLP architecture |
| Graph Neural Reasoning (GNN) | Kipf & Welling (2017), Schlichtkrull et al. (2018) | Extensively validated |
| Learner State Update & Forgetting | Zhou & Wang (2025) – *Scientific Reports* | Implemented in real WeChat learning system |
| Gap Detection & RL-inspired Path | Reinforcement-based recommendation studies | Supported by literature |
| KGT-based Personalization | Sun et al. (2024, Duke University) | Concept experimentally validated |

Thus, this chapter defines a **feasible engineering model** that integrates these existing paradigms into a cohesive learner-centric framework.

## **3.3 System Overview** {#3.3-system-overview}

The proposed model represents a hybrid neural–graph framework designed to simulate and visualize a learner’s linguistic knowledge evolution. Figure 1 illustrates the overall architecture. The system processes learner interactions, transforms linguistic inputs into semantic embeddings, integrates them into a dynamic graph, and continuously updates the learner’s personalized knowledge graph.

**Main components:**

1. **Learner Interaction Data** – textual or task-based input collected from reading, writing, or comprehension exercises.  
2. **Semantic Embedding Layer (BERT/K-BERT)** – transforms raw text into contextual vector representations while incorporating prior knowledge.  
3. **Graph Integration Layer (GNN)** – updates node embeddings by propagating information through linguistic relationships.  
4. **Learner State Update** – adjusts mastery scores, accounts for propagation and forgetting.  
5. **Gap Detection & Adaptive Path Recommendation** – identifies weak nodes and suggests optimal learning paths.

**Figure 1\. System Overview Diagram**

| \[ Learner Input \] → \[ BERT/K-BERT Embedding \] → \[ Graph Integration (GNN) \] \-  → \[ Learner Graph Update \] → \[ Adaptive Feedback \] |
| :---- |

## **3.4 Semantic Embedding and Knowledge Injection** {#3.4-semantic-embedding-and-knowledge-injection}

The **semantic embedding layer** employs transformer-based language models such as BERT or K-BERT to encode linguistic elements. K-BERT enriches representations by injecting knowledge triples (subject–relation–object) from a predefined linguistic knowledge graph. This injection allows the model to contextualize tokens not only syntactically but also semantically.

Mathematically, a sentence is embedded as:

Each token’s embedding is compared to knowledge node embeddings to identify relevant nodes:

This stage operationalizes the **semantic alignment** between language usage and structured linguistic knowledge. It can be realized using pretrained models or lightweight domain-specific fine-tuning.

**Figure 2\. Semantic Embedding and KG Injection**

| \[ Sentence Tokens \] → \[ Contextual Embedding \] \+ \[ Knowledge Triples Injection \] \-   → \[ Enriched Representation \] |
| :---- |

## **3.5 Graph Neural Reasoning Layer** {#3.5-graph-neural-reasoning-layer}

To simulate how linguistic knowledge propagates across related concepts, the model incorporates a **Graph Neural Network (GNN)**. Each node represents a linguistic concept (e.g., tense, part-of-speech), and edges represent prerequisite or semantic relations. This layer mimics the cognitive phenomenon of association — when mastering one rule strengthens related rules.

Information propagation between nodes is defined as:

where are neighbors under relation type , and is a non-linear activation function.

**Figure 3\. Graph Reasoning Flow**

![][image1]

## **3.6 Learner Knowledge State Update** {#3.6-learner-knowledge-state-update}

Each learner has a knowledge vector , where each indicates mastery of node . After each interaction, mastery is updated based on performance and graph propagation.

**![][image2]**

Typical parameters: . These values are drawn from empirical observations in prior reinforcement-based learning systems (Zhou & Wang, 2025).

**Figure 4\. Learner State Update Mechanism**

| \[ Learner Response \] → \[ Direct Update \] → \[ Graph Propagation \] \- → \[ Forgetting Function \] → \[ New State s(t+1) \] |
| :---- |

## **3.7 Gap Detection and Adaptive Path Recommendation** {#3.7-gap-detection-and-adaptive-path-recommendation}

After each update, the system identifies weak areas (low mastery nodes) and suggests the next learning targets. The gap score for each node is defined as:

where is the difficulty or pedagogical weight of the node.

Candidate nodes are filtered by prerequisite feasibility and ranked by gap magnitude. A reinforcement-inspired rule ("prune–then–select") determines which concepts to recommend next. This mirrors reinforcement learning’s policy optimization but is designed to operate without full model training, making it lightweight and theoretically plausible.

**Figure 5\. Gap Detection and Recommendation Flow**

| \[ Learner Graph \] → \[ Compute Gap Scores \] → \[ Prune Invalid Nodes \] → \[ Select Next Concepts \] → \[ Update Learning Path \] |
| :---- |

## **3.8 Personalization via Knowledge Graph Tuning (KGT)** {#3.8-personalization-via-knowledge-graph-tuning-(kgt)}

Instead of retraining model parameters, personalization is achieved through **Knowledge Graph Tuning (KGT)** — modifying edges and weights in the learner’s personal graph. When consistent feedback indicates misunderstanding or preference, the graph adapts accordingly:

where and represent edge addition and removal. This dynamic updating process improves computational efficiency and interpretability. Teachers and learners can visualize these modifications to understand why certain topics are prioritized.

**Figure 6\. KGT-Based Personalization**

![][image3]

### 

## **3.9 Theoretical Feasibility and Expected Implementation Path** {#3.9-theoretical-feasibility-and-expected-implementation-path}

This methodology is technically realizable using existing machine learning frameworks. Implementation could be achieved in three incremental phases:

1. **Prototype Simulation:** Use synthetic learner data to simulate graph evolution and verify the correctness of update rules.  
2. **Integration with Subturtle Platform:** Map real learner interaction logs to the framework’s data schema (inputs, responses, timestamps).  
3. **Reinforcement Adaptation:** Gradually incorporate reinforcement learning components for path optimization.

The overall design, while untested, is **computationally plausible** and grounded in current research trends, ensuring that future implementation would not require unproven technologies but only integration of existing modules.

## 

## 

# **References** {#references}

1. Abu-Salih, B., & Alotaibi, S. (2024). A systematic literature review of knowledge graph construction and application in education. Heliyon, 10(3), Article e25383. [https://doi.org/10.1016/j.heliyon.2024.e25383](https://doi.org/10.1016/j.heliyon.2024.e25383)

2. Balog, K., & Kenter, T. (2019). Personal knowledge graphs: A research agenda. Google Research. [https://research.google/pubs/personal-knowledge-graphs-a-research-agenda/](https://research.google/pubs/personal-knowledge-graphs-a-research-agenda/)

3. Dessì, D., Osborne, F., Reforgiato Recupero, D., Buscaldi, D., & Motta, E. (2021). Generating knowledge graphs by employing natural language processing and machine learning techniques within the scholarly domain. Future Generation Computer Systems, 116, 253–264. [https://doi.org/10.1016/j.future.2020.10.026](https://doi.org/10.1016/j.future.2020.10.026)

4. Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., & Wang, P. (2019). K-BERT: Enabling language representation with knowledge graph. arXiv preprint arXiv:1909.07606. [https://doi.org/10.48550/arXiv.1909.07606](https://doi.org/10.48550/arXiv.1909.07606)

5. Sun, J., Du, Z., & Chen, Y. (2024). Knowledge graph tuning: Real-time large language model personalization based on human feedback. arXiv preprint arXiv:2405.19686. [https://doi.org/10.48550/arXiv.2405.19686](https://doi.org/10.48550/arXiv.2405.19686)

6. Zhou, L.-Y., & Wang, Y.-Y. (2025). Simulation of personalized English learning path recommendation system based on knowledge graph and deep reinforcement learning. Scientific Reports, 15, Article 17918\. [https://doi.org/10.1038/s41598-025-17918-x](https://doi.org/10.1038/s41598-025-17918-x)

7. Nongkhai, L. N., Wang, J., Wynn, A., & Mendori, T. (2026). Evaluating adaptive and generative AI-based feedback and recommendations in a knowledge-graph-integrated programming learning system. Computers and Education: Artificial Intelligence, 10, Article 100526\. [https://doi.org/10.1016/j.caeai.2025.100526](https://doi.org/10.1016/j.caeai.2025.100526)

8. Lou, L. L. (2025). From Fragmentation to Coherence: A Knowledge Graph-Driven Integration Framework for Blended Learning. Arab World English Journal, 16(3). [https://doi.org/10.24093/awej/vol16no3.12](https://doi.org/10.24093/awej/vol16no3.12)

9. Zhang, C. (2025). Optimising AI writing assessment using feedback and knowledge graph integration. PeerJ Computer Science, 11\. [https://doi.org/10.7717/peerj-cs.2893](https://doi.org/10.7717/peerj-cs.2893)

10. Zhu, T. (2025). An Intelligent Guide Application for English Online Education Based on Deep Learning. Journal of Advanced Computational Intelligence and Intelligent Informatics, 29(3), 489-499. [https://doi.org/10.20965/jaciii.2025.p0489](https://doi.org/10.20965/jaciii.2025.p0489)

11. Duan, C., Yang, J., & Zhang, M. Y. (2025). Enhancing the Recommendation of Learning Resources for Learners via an Advanced Knowledge Graph. Applied Sciences-Basel, 15(8). [https://doi.org/10.3390/app15084204](https://doi.org/10.3390/app15084204)

12. Yu, G. F., Xie, Z. W., & Huang, J. X. (2025). Exploring long- and short-term knowledge state graph representations with adaptive fusion for knowledge tracing. Information Processing & Management, 62(3). [https://doi.org/10.1016/j.ipm.2025.104074](https://doi.org/10.1016/j.ipm.2025.104074)

13. Li, C. H., & Luo, Y. (2025). Integrating Knowledge Graph Reasoning and Reinforcement Learning for Explainable MOOC Recommendations. IEEE Access, 13, 183722-183733. [https://doi.org/10.1109/access.2025.3625213](https://doi.org/10.1109/access.2025.3625213)

14. Zhang, Z. W. (2025). Dual-Channel Knowledge Tracing With Self-Supervised Contrastive and Directed Interaction Learning. IEEE Access, 13, 32276-32288. [https://doi.org/10.1109/ACCESS.2025.3542870](https://doi.org/10.1109/ACCESS.2025.3542870)

15. Zong, C. M., Liu, Y. R., & Wu, L. L. (2025). DPDG-GKT: A Dual-Path Dynamic Gated GNNs Model for Knowledge Tracing. IEEE Access, 13, 198826-198835. [https://doi.org/10.1109/access.2025.3635900](https://doi.org/10.1109/access.2025.3635900)

16. Cheng, K. W., Ahmed, N. K., & Sun, Y. Z. (2024). Neural-Symbolic Methods for Knowledge Graph Reasoning: A Survey. ACM Transactions on Knowledge Discovery from Data, 18(9). [https://doi.org/10.1145/3686806](https://doi.org/10.1145/3686806)

17. Zhang, Y. C., Kong, X. J., & Dong, B. (2024). A survey on temporal knowledge graph embedding: Models and applications. Knowledge-Based Systems, 304\. [https://doi.org/10.1016/j.knosys.2024.112454](https://doi.org/10.1016/j.knosys.2024.112454)

18. Jia, N. N., & Yao, C. Y. (2024). ShallowBKGC: a BERT-enhanced shallow neural network model for knowledge graph completion. PeerJ Computer Science, 10\. [https://doi.org/10.7717/peerj-cs.2058](https://doi.org/10.7717/peerj-cs.2058)

19. Xia, X. N., & Qi, W. X. (2024). Multilayer knowledge graph construction and learning behavior routing guidance based on implicit relationships of MOOCs. Technological Forecasting and Social Change, 204\. [https://doi.org/10.1016/j.techfore.2024.123442](https://doi.org/10.1016/j.techfore.2024.123442)

20. Cui, C. R., Yao, Y. M., & Ko, J. (2024). DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing. ACM Transactions on Information Systems, 42(3). [https://doi.org/10.1145/3696415](https://doi.org/10.1145/3696415)

21. Shang, B., Zhao, Y. L., & Liu, J. (2024). Knowledge graph representation learning with relation-guided aggregation and interaction. Information Processing & Management, 61(4). [https://doi.org/10.1016/j.ipm.2024.103740](https://doi.org/10.1016/j.ipm.2024.103740)

22. Liu, S. Y., Liu, S. Y. J., & Du, S. H. (2024). Heterogeneous Evolution Network Embedding with Temporal Extension for Intelligent Tutoring Systems. ACM Transactions on Information Systems, 42(2). [https://doi.org/10.1145/3617828](https://doi.org/10.1145/3617828)

23. Dong, H., Wang, P. Y., & Zhou, Y. C. (2024). Temporal inductive path neural network for temporal knowledge graph reasoning. Artificial Intelligence. [https://doi.org/10.1016/j.artint.2024.104085](https://doi.org/10.1016/j.artint.2024.104085)

24. Shang, B., Zhao, Y. L., & Liu, J. (2024). Learnable convolutional attention network for knowledge graph completion. Knowledge-Based Systems. [https://doi.org/10.1016/j.knosys.2023.111360](https://doi.org/10.1016/j.knosys.2023.111360)

25. Cheng, K., Peng, L. Z., Wang, P., Ye, J. C., Sun, L. L., & Du, B. W. (2024). DyGKT: Dynamic Graph Learning for Knowledge Tracing. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2024 (pp. 409-420). [https://doi.org/10.1145/3637528.3671773](https://doi.org/10.1145/3637528.3671773)

26. Liu, H. B., Chen, Y., & Zhang, J. E. (2023). An inductive knowledge graph embedding via combination of subgraph and type information. Scientific Reports, 13(1). [https://doi.org/10.1038/s41598-023-48616-1](https://doi.org/10.1038/s41598-023-48616-1)

27. Jiang, D., Wang, R. G., & Yang, J. (2024). Multisource hierarchical neural network for knowledge graph embedding. Expert Systems with Applications, 237\. [https://doi.org/10.1016/j.eswa.2023.121446](https://doi.org/10.1016/j.eswa.2023.121446)

28. Sola, F., Ayala, D., & Ruiz, D. (2023). Deep embeddings and Graph Neural Networks: using context to improve domain-independent predictions. Applied Intelligence, 53(19), 22415-22428. [https://doi.org/10.1007/s10489-023-04685-3](https://doi.org/10.1007/s10489-023-04685-3)

29. Hoang, V., Jeon, H. J., & Lee, O. J. (2023). Graph Representation Learning and Its Applications: A Survey. Sensors, 23(8). [https://doi.org/10.3390/s23084168](https://doi.org/10.3390/s23084168)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbIAAABwCAYAAAB7JPUXAAAVBklEQVR4Xu2dd5MVRRfG3y9AEhCUjAhKkFRIUEklQZJEQQmWWAoWSLJARECiUpYgKAISFJBQgAEli4AKK0WOSkagXKLfYd56uqqnek73zJ3de3dg1uePX+30Oad7esKdZzrs9P/+/fdfjxBCCEkr/5MGQgghJE1QyAghhKQaChkhhJBUQyEjhBCSaihkhBBCUg2FjBBCSKqhkBFCCEk1FDJCCCGphkJGCCEk1VDICCGEpBoKGSGEkFSTUchWr17tk5eXZ/kLwunTp70SJUoo6tWrZ/kJIYSQgpJRyCA6P/74o2LhwoUq3aFDBysuDsgrbYQQQkg2xBIyM92/f3/LFodbt24VKh8hhBASRYGFrE+fPpZt6dKlfpchWLZsmVWGpH79+qExs2fPtuoxfPhwb+PGjV7jxo39uHbt2sWug7mPpk2bWj5CCCHpJJaQmQwdOjTgr1Wrlle6dGkrjyzn5s2bTrsrHukXXnghYIOQwZ6fn+/b7t69q/7euHHDWYdu3bpZNgoZIYQUL2IJ2c6dOxWLFy+2RAbpDRs2eLdv31bdh2DSpEne9u3bA+WECZnL3qNHD8sGIevZs6eVH6C7U9dB1wN1kGUcPnxYceLECasMQggh6SSWkJnpAwcOKNuVK1d8v4vJkycH8rkEC2zevNmyT5061bJByDDZROYH5cuXt/avkbGEEEKKFwUWMm1btGiRv41WjoyRhAnZ3r17Lfs777xj2fQYmcwP0L0Zpw6EEEKKHwUWsl9//VXZDh065Ptr165t5ZOECZkuQ6YfffTRgC1KyPCvAXHqoFtpHCMjhJDiQywhGzZsmKJr164q/fDDD/t+/U/OzZo1U2KFrkLMLJTlRAlZx44dverVq3unTp3yvv32WxV39erVQEyUkAFdh4MHD/p1wFibjKGQEUJI8SKWkGnwNY5NmzZZMRCpSpUq+XHNmzd3xoQJGShXrpyf/9y5c5Y/k5C56iC7GylkhBBS/MgoZIQQQsiDDIWMEEJIqqGQEUIISTUUMkIIIamGQkYIISTVUMgIIYSkGgoZIYSQVEMhI4QQkmoSFzJ8F1HaTPD1+nXr1imkL1vwz9BY8kXakwbHuGPHDsv+ILJlyxZ1Lc6cOWP5kgSfIdPL9hSUwp7rorgHi5KBAwdm/H1lorDnKheUKlXKmzNnjmUnJBOJClmbNm3UEivSrsH6YeaXRKQ/W7CWGr4gIu1Jc+nSpayOr1WrVoHzNGPGDCsmF5j7cC1UmiSNGjVSX2/BdtS5c/lcNs3MmTNDX26i8j2IoL5Rv6843M9jPn78+H3dP0kviQnZxYsXM96k8B89etSy5xLs4+uvv7bsaQEPdPkNyU8//dSKy5a5c+dmvF5JEkfI0Grr1auXZY8CZZ0/f96ya5+0PaiULVvWe+655yx72qhQoYK61tJOSBSJCVmfPn1CHwxYuwzAP27cOD9txuAh1r59e9VqwyKasoy8vDw/z4ABA9TX8LEcjIzDV/Wxfpm0J8XHH3/sI30A3Y7oImrYsKFqQV6/ft333bt3L/Qcmvz222/eU0895b322mven3/+GfDhHO3Zs0d1FeLbmaNGjQr4EY+Ydu3aqX3pa4F16HQMWjFmnt27d6tVEUwbXki6d+/utWzZ0ps2bVrAB3AsEydO9OrWreutWLHC8gPk79evn6pTHCGbMmVKYM26qHMd557T+8FHqHE9lixZYpWTDbnsukRdo7oFp0+f7j322GNqUVx0F5u+Y8eORZ4r5J09e7Y6D+hVefnll60YgGvat29fda4mTJhg+devX69ewkaMGKFipR+gazHs+hISRmJChpvT9TFh0LlzZwViIFY6rf166Rg82PQDFj8Ws4yVK1cq+6BBg1Rfe4sWLZw/CDw8XfakQLfg008/7ayDbrXWrFlTCUydOnUCKw0sX77cmc9k8ODBKuaNN97wnnjiCSseaQh86dKl1aoDSGM/2n/kyBF17nVefS3Q2jHLMMucNWuWN2/ePD8NoUMMyh8yZIhXpkwZ78MPP7TqUaVKFW/06NH+MUs/rvErr7yitvEQ1kLWpEkTJfg67u2331bbnTp1CnxwGudad8OaZYNM95wu+/333/dq1Kjh1a9fX6U///xzq6zCgvL0OZa+guI6Rg2uNfzvvfeef81N/7Zt2yLPFVpJVatWVecBL1mu8/DNN98oe+vWrdULmHnfgmrVqik/hAzXGtuuj4P/888/yqevLyFxSFTIxo4da9lljKubB/YxY8ZYNjOthWz+/PlWfpOtW7daeZMmbIwsTOA0Xbp0ifSfOHFC+SGI2oZlccwJAPBjuRudxtu4q0y8UbvsugwzLYUM/qiHM1YpuHDhglWmvsZaNKQ/Pz9fbS9evFiJ+p07d5Rdx8o8Zl5pM32ue0771q5d66f1S4KMyxa0uvVx4NxIfxzC6oXlkOALGweUuMqBkEm7mUbrHmm9RqEE43Z4uTRtP/30k1WmWTbuW2knJIxEhUx2Y0nCHiqwnz171rKZaS1kMq8E3SNx4iT6QeMik/+zzz4LlBUmZGvWrFF2PMivXbtm+Xv37u3Mp0G3mstv2rA9derUUL8mGyFDyxsx/fv3d3YhyfOj0Q9xbKPb08xTsmRJX8gg1FjWB8drip6sl7k/aTN9rntO+8yZkrqliW0sESTrb6LzhyH3BbAv9CiE+aOIyqP3iS5g6ZO4yoGQoVUXFofuSFc+DbpQ5fFHnQfY8TuVdkLCSFTI0DUo7TJGPlQuX77svOGlLa6QocUWJ06CrpMwMvkxG8ssK0zIALrwKlas6P/QzTGfTOMHI0eOdPpNG7ZN0ZF+TTZCBtAlp48BmN1ISOOh6kL7ZYvuySef9IUMYE05tCzxkoD4sPtElydtpk/ec2H5cJzahlaUvM4FuSfkvvbv368Wl0X5GAeW/kzIuppAIHWXIoiaou8qR3cthsVhzMyVTzN+/Hj1UiOvdZiwoiw5tktIFIkJmWu8RhL2UIHdtUimmY4rZJjZFSeuKIkSMpMGDRoE4nQrQMZpVq1a5fSbNmxL0XHlKYiQoXWEWY4yDugFVWUdZJwJ/K4xM1PIdJl4SKOlilZmWLlhdu1z3XOufKaQ5Qq0wvUYFmZcortUxsQB+XFfSbsEXaVRx+DyZRKypUuXOvNptm/froRU2l189913kWUR4iIxIcNDKNMNGvZQ0Ss/6zQmK8iy4goZYu53t0WYkH3xxReBtOvhjJaKfKPG2I3eRjwmL+g0BAHjgqY/F0KGAX1s65mUppDJ2XOYJICuQZ1G60MKFc6J/teLvXv3Bvat0y4hw7a+t6LqK22mDxNjpN2VL9dChrIwG1DaC0PlypWt7ljw119/Bf63DMIfdQwuXyYh02m0KGVe04/rbtpcx46xtLZt21p2QqJITMiAvPkl8LuEzHxQaeQ04ThChmn7mWKKEszokscBMG0cfjwspE+KG5Ax5g8fM9CkX+bNVsj0WJ1Gdi3K/QPz3wgAHlgyBoKl/XrWqcacfq/3Yc6CRdocxzFbghKzHpjJF+aT6VwLWS7BbFNX3f744w/r+OW/n0i/pkOHDsofR8j0pBIT06/vJ4kZo8s1rzMhcUhUyDCrDtPfpT0umzZtUg9qaY8LfiRy4sWDBqYfY3A803FiFtqCBQu8K1euWD6AmX1hs8hyAcYwDh48aNk1J0+eVF8DkW/hJnhg4X/I8D+A0gfQstu3b59lJ27wT9Fhvy+cS7zsxel+zAa8jOD/7cyZsya4H/A7dn2BBP/jhrFQaSckE4kKGSGEEJJrKGSEEEJSDYWMEEJIqqGQEUIISTUUMkIIIamGQkYIISTVUMgIIYSkGgoZIYSQVEMhI4QQkmooZIQQQlINhYwQQkiqoZARQghJNRQyQgghqYZCRgghJNVQyAghhKQaChkhhJBUQyEjhBCSaihkhBBCUg2FjBBCSKqhkBFCCEk1FDJCCCGphkJGCCEk1VDICCGEpBoKGSGEkFRDISOEEJJqKGSEEEJSDYWMEEJIqsm5kO3evdtbt26dZSeEEEKKgthCVqJECYW0u2JccTNnzvQWLlxo2QkhhJBsiCVkPXr0CBUoTV5enjdo0CDLrkHeRo0aWXZCCCEkG2IJGUTo3Xff9dq0aeOdP38+4Js+fbo3efJkb/DgwV6rVq3UNtD+FStWqDTKqFq1qu///fffA+WsX79eCWa7du28e/fuWXXYsGGDd+TIEe/WrVtet27dvHr16nk3btwIxHzwwQcKmZcQQkjxJbaQ3b592/vll1+8Xr16BXwvvvii17lzZ6958+ZerVq11DbQ/okTJ6o0yihfvrzv/+GHH/yYXbt2KT+ErG/fvmr73Llzgf0MHz7cW716tV9O48aNvZYtW1r1jGo1EkIIKX7EFjLXtsn333+vxEbazXxhXYuyTLT8pA1lw5afn+/b7t69a5Uj8xFCCCneZBSy2bNneyNGjPDTEIqff/7ZiiuskGGGoxYgiRmHstG9KfMTQgj5b5NRyEqWLOkNHDjQe/311xUQGHTrybjCCtn48eOVD9P2JWYcysY4mcxPCCHkv01GIYPIVKlSxadcuXJWawkUVsi2b9/uLE+Csjdu3GjZCSGE/LeJFLI333zTKTKwLVu2LGDLJGTPP/+8syxQqlQpb//+/QEbujTNdBwhc3VJEkIIKd5EChlEoWzZsk57xYoVA7ZMQoaJGVpowJIlS3zfzZs3Az6XIFHICCGEuIgUsqRBq+yzzz5T/ysmfYQQQoiLB0rICCGEkIJCISOEEJJqKGSEEEJSDYWMEEJIqqGQEUIISTUUMkIIIamGQkYIISTVJC5kDRs2tGxJcT/3TUi2rF271jt58qRlTwv8/ZGiIlEhw3pm8+fPt+waLJT50EMPFdkXOkqXLu1NmjTJsj8oTJkyxT921FX6c02mc7x48WK/PvKTZACrImQqIw5YrFWXgzXn8JUYGaN9a9asseyFqcNHH31UqHz3k2zri/xY/Fbak6J+/fpe+/btLTsh2ZKokGX6IcJft25d7+LFiwrpzxYIaaY6PCjcbyHD+Ycfq3JjG58RkzFFIWRhH6UGsHfs2NFplzagP30m7SBtQjZt2rSs63u/hSzqehCSDYkJ2dy5czPexPDLlaFzDfbRtWtXy/6gkYSQRYG13zJdr1xhCtmECRNC9wt7pu9tmkQ9ONMmZKjr3r17LXvaqFatGj9BR3JOYkKGJWDCHs74kbowY1q0aBHwXb9+PeBfuXKlsuNHElYG6Ny5s9OeFKi3Wb9mzZpZMSDsXMFu5keLScbo4zPjNm/eHPCHnZ99+/YF/Jrly5f7MX///XdkGQDdSGbM7du3A37Td+bMGb+cXbt2+dtfffWVum/Onj3r59HlZKqD6XPFaSGrXLmy79u2bZtVTmE5deqUKhMvcNJXUA4cOOA8Rs1bb70VOMZPPvkk4B89erTvc7XIYDfvy7B7z9wHWs6mb8+ePQH/uHHjrPwALcvp06dbdkKyITEhw83du3dvyy5j8HYu7d26dVM+na5Zs2YgDbSQAd0NtnXrVqushQsXWnmTBPuGWOh0hQoVrBgQ9jAxzyEe6q5jgQ1vvlOnTlXpo0ePOlsyrryaqJaRJqxrEeOcZcqU8e7cuaPSzzzzTCAOi6aaD0J93cw0/uKhi20svmraJWH2OC0yCCfSeXl5obGFZdiwYf6xzZkzx/LHZeTIkaF1cx3jl19+acUBxIUJGa4Xti9fvqzSqLuMMcXJ7OI9duyY8s+aNUuljx8/btVJg3sRwwfSTkg2JCpkY8eOtewyxiVksI8ZM8aymWktZFGTSQDETeZNigYNGsTed5iQSVDevHnzLNuAAQOsWElUXbIRMtju3btn2cztq1ev+mmIs/SjFYa/ixYt8sXeta8ou+shr3F1Lcp0rkALBGWDmTNnWv5MQDTC6rZu3bpQnwRxYUJm/u4ef/zxwP23YsWKyH3AJ3sWMKnKtaK7Xs5J2gnJhkSFbMiQIZZdxkgh012Frq4pM62FTJYpWbVqVaw4iX4Qucjkx9I0OqZfv35W2S7ChKxDhw5W+bIbR9cpE1Fx2QqZC9Nvxh88eNDyY2FVbdN/5Rp4YeVpshEyWXeTw4cPK6TdRO4PmN2h0hfF4MGDI/PUrl3bL7dnz56WXwN/mJCZadkD0rhxYytG5neBxXRl7IULF7xKlSpZdkKyIVEhw7iJtMsYKWTafu3aNctmpuMK2ahRo2LFFQXYb6dOnSy7C5eQuSbMIC1bujImjKi4bIVM2qL8mMRg2tAi6NWrlxof0/H414yBAwdaZbnK02QjZLkEXaMoGyxYsMDyZwJdhZnqhhasHisLe1mCrzBC1rZtWytG5tcva5lACzKsfoQUlsSE7NVXX438MQD4w4QMA//SZqbjChlWvMYAv7QnAcaFSpYsadlduI6lUaNGXqlSpay4tAsZxvJMGx7IeGvX3XDdu3f3hg4dqrq4ZFmu8jRR3VhFLWTY90svvaTKBOgilTEFAWXICU5hhB0H7IURMlwHGSPz41il3QV6FLCArrQTkg2JCRmI+jFov0vI9IMOA/I7d+70Hw5mTBwh05Mj5PhNkmD/GL/CTD3ZEpFxbdq0Ua0wPZ6EMQfY8faNAXY96aUgQoYZcDt27FAgTm+fPn06EBclZDpP3759A2VoP8ZLYMfYHf4HTU6w0TPktmzZouqDiQamHxMCzDRaY7Iuep/yOGRd4cPLC86dOWZT1EKGfyM5dOiQZS8smLX7yCOPWHac4yZNmqjfzaVLl7ymTZt6/fv3D8SY5wlf15DnSh63FDJQo0YNr2rVquqexYxMs8cAk3oQDz+uLX6jmB1s5g/bFyG5IFEhwwP8ypUrll2Dm9wlZGDp0qXKD9BnL/1xhKxLly5e9erVLXvS6Ac3CGtl4GGhBcF8IGL2m86rp1UXZIysdevWfn4TfFXEjIsSMplXY8bgSyCmDzPvTL/57xTm/5GZ+8iUdmHGALy04P8Gpb+ohSzX5OfnO+uHlzvMAtTH5xp/kudIngtZrkvIAGbC6rzoopf+OnXq+H6M20n/jBkznOUSki2JChkIG7BPAvyI9JRwQtLGs88+a9nSBH5/eEGTdkKyJXEhI4QQQnIJhYwQQkiqoZARQghJNRQyQgghqYZCRgghJNVQyAghhKQaChkhhJBUQyEjhBCSaihkhBBCUg2FjBBCSKr5P2zRjxGIYk3TAAAAAElFTkSuQmCC>