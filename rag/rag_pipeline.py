"""
rag/rag_pipeline.py - Full RAG pipeline
Handles: knowledge base loading, chunking, embedding, vector store, retrieval
"""
import os
import json
from typing import List, Tuple
from config import config


# ─────────────────────────── Knowledge Base Documents ──────────────────────
KNOWLEDGE_BASE = [
    {
        "id": "alg_001",
        "title": "Quadratic Formula",
        "topic": "algebra",
        "content": """Quadratic Formula: For ax² + bx + c = 0, solutions are x = (-b ± √(b²-4ac)) / 2a.
Discriminant D = b²-4ac. D>0: two real roots. D=0: one repeated root. D<0: complex roots.
Sum of roots = -b/a. Product of roots = c/a. Always check if factoring is easier first."""
    },
    {
        "id": "alg_002",
        "title": "Factoring Techniques",
        "topic": "algebra",
        "content": """Factoring Techniques:
1. Common factor: ax + ay = a(x+y)
2. Difference of squares: a²-b² = (a+b)(a-b)
3. Sum/difference of cubes: a³±b³ = (a±b)(a²∓ab+b²)
4. Perfect square trinomial: a²+2ab+b² = (a+b)²
5. Grouping: for 4-term polynomials, group pairs.
JEE tip: Always try rational roots ±(factors of constant)/(factors of leading coeff) first."""
    },
    {
        "id": "alg_003",
        "title": "Inequalities",
        "topic": "algebra",
        "content": """Inequality Rules:
- Adding/subtracting same value: inequality direction unchanged.
- Multiplying/dividing by POSITIVE: unchanged. By NEGATIVE: FLIP inequality.
- For |x| < a: -a < x < a. For |x| > a: x < -a or x > a.
- AM-GM inequality: (a+b)/2 ≥ √(ab) for non-negative a,b. Equality when a=b.
- Cauchy-Schwarz: (a₁b₁+a₂b₂)² ≤ (a₁²+a₂²)(b₁²+b₂²)"""
    },
    {
        "id": "prob_001",
        "title": "Probability Fundamentals",
        "topic": "probability",
        "content": """Probability Rules:
- P(A) = favorable outcomes / total outcomes (classical definition)
- 0 ≤ P(A) ≤ 1 always
- P(A') = 1 - P(A) (complement rule)
- P(A∪B) = P(A) + P(B) - P(A∩B) (addition rule)
- P(A∩B) = P(A)·P(B|A) = P(B)·P(A|B) (multiplication rule)
- Independent events: P(A∩B) = P(A)·P(B)
- Bayes theorem: P(A|B) = P(B|A)·P(A) / P(B)"""
    },
    {
        "id": "prob_002",
        "title": "Permutations and Combinations",
        "topic": "probability",
        "content": """Counting Principles:
- Permutation (order matters): P(n,r) = n! / (n-r)!
- Combination (order doesn't matter): C(n,r) = n! / (r!(n-r)!)
- Circular permutation: (n-1)! ways to arrange n objects in a circle
- Repetition allowed permutation: nʳ
- With identical objects: n! / (p!q!r!...) where p,q,r are counts of identical items
- Pascal's identity: C(n,r) = C(n-1,r-1) + C(n-1,r)
- Binomial theorem: (a+b)ⁿ = Σ C(n,k) aⁿ⁻ᵏ bᵏ"""
    },
    {
        "id": "prob_003",
        "title": "Conditional Probability and Bayes",
        "topic": "probability",
        "content": """Conditional Probability:
P(A|B) = P(A∩B) / P(B), provided P(B) ≠ 0.
Total Probability Theorem: P(A) = Σ P(A|Bᵢ)·P(Bᵢ) where Bᵢ are mutually exclusive exhaustive events.
Bayes Theorem: P(Bᵢ|A) = P(A|Bᵢ)·P(Bᵢ) / Σ P(A|Bⱼ)·P(Bⱼ)
Common mistake: P(A|B) ≠ P(B|A). Always identify which direction you need."""
    },
    {
        "id": "calc_001",
        "title": "Limits",
        "topic": "calculus",
        "content": """Limits:
- lim(x→a) f(x) = L means f(x) approaches L as x approaches a.
- Standard limits: lim(x→0) sin(x)/x = 1, lim(x→0) (1-cos x)/x² = 1/2
- lim(x→∞) (1+1/x)ˣ = e, lim(x→0) (1+x)^(1/x) = e
- L'Hôpital's rule: if 0/0 or ∞/∞ form, lim f/g = lim f'/g'
- Squeeze theorem: if g(x) ≤ f(x) ≤ h(x) and lim g = lim h = L, then lim f = L
- For polynomials at ∞: divide numerator and denominator by highest power of x."""
    },
    {
        "id": "calc_002",
        "title": "Derivatives",
        "topic": "calculus",
        "content": """Derivative Rules:
- Power rule: d/dx(xⁿ) = nxⁿ⁻¹
- Product rule: d/dx(uv) = u'v + uv'
- Quotient rule: d/dx(u/v) = (u'v - uv') / v²
- Chain rule: d/dx(f(g(x))) = f'(g(x))·g'(x)
- Common: d/dx(sin x) = cos x, d/dx(cos x) = -sin x
- d/dx(eˣ) = eˣ, d/dx(ln x) = 1/x
- d/dx(aˣ) = aˣ ln a, d/dx(logₐx) = 1/(x ln a)
- Implicit differentiation: differentiate both sides, isolate dy/dx."""
    },
    {
        "id": "calc_003",
        "title": "Integration",
        "topic": "calculus",
        "content": """Integration Techniques:
- Power rule: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ -1)
- ∫sin x dx = -cos x + C, ∫cos x dx = sin x + C
- ∫eˣ dx = eˣ + C, ∫1/x dx = ln|x| + C
- Substitution: let u = g(x), du = g'(x)dx
- Integration by parts: ∫u dv = uv - ∫v du (choose u by LIATE: Log, Inverse, Algebraic, Trig, Exponential)
- Definite integral: ∫[a,b] f(x)dx = F(b) - F(a) (Fundamental Theorem of Calculus)
- Area between curves: ∫[a,b] |f(x) - g(x)| dx"""
    },
    {
        "id": "calc_004",
        "title": "Optimization",
        "topic": "calculus",
        "content": """Optimization (Finding Max/Min):
1. Find f'(x) and set = 0 to find critical points.
2. Second derivative test: f''(x) > 0 → local min, f''(x) < 0 → local max.
3. Check boundary points for absolute max/min on closed interval.
4. For word problems: define variable, write objective function, find domain, differentiate.
Common JEE types: area/perimeter optimization, container volume maximization, profit/cost problems.
AM-GM shortcut: for positive quantities, min of (x + k/x) = 2√k at x = √k."""
    },
    {
        "id": "linalg_001",
        "title": "Matrices and Determinants",
        "topic": "linear_algebra",
        "content": """Matrix Operations:
- Addition: element-wise. Multiplication: (AB)ᵢⱼ = Σ aᵢₖ bₖⱼ
- AB ≠ BA in general (not commutative)
- Determinant 2×2: |A| = ad-bc. 3×3: expansion along any row/column.
- |AB| = |A|·|B|. |Aᵀ| = |A|. |kA| = kⁿ|A| for n×n matrix.
- Inverse: A⁻¹ = adj(A)/|A|. Exists iff |A| ≠ 0.
- Properties: (AB)⁻¹ = B⁻¹A⁻¹. (Aᵀ)⁻¹ = (A⁻¹)ᵀ
- Cramer's rule: xᵢ = Dᵢ/D for system of linear equations."""
    },
    {
        "id": "linalg_002",
        "title": "System of Linear Equations",
        "topic": "linear_algebra",
        "content": """Systems of Linear Equations (Ax = b):
- Unique solution: |A| ≠ 0 (Cramer's rule or row reduction)
- No solution (inconsistent): rank(A) < rank(A|b)
- Infinitely many solutions: rank(A) = rank(A|b) < n
- Gaussian elimination: use elementary row operations to get row echelon form.
- Gauss-Jordan: reduce to reduced row echelon form (RREF).
- Homogeneous Ax=0: always has trivial solution x=0. Non-trivial solution iff |A|=0."""
    },
    {
        "id": "linalg_003",
        "title": "Eigenvalues and Eigenvectors",
        "topic": "linear_algebra",
        "content": """Eigenvalues and Eigenvectors:
- Av = λv, where λ is eigenvalue and v is eigenvector.
- Find eigenvalues: solve characteristic equation det(A - λI) = 0.
- For each λ, find eigenvector: solve (A - λI)v = 0.
- Trace = sum of eigenvalues. Determinant = product of eigenvalues.
- Symmetric matrices have real eigenvalues.
- Diagonalization: A = PDP⁻¹ where D has eigenvalues on diagonal.
- Cayley-Hamilton theorem: every matrix satisfies its own characteristic equation."""
    },
    {
        "id": "common_001",
        "title": "Common Mistakes in JEE Math",
        "topic": "general",
        "content": """Common Mistakes to Avoid:
1. Dividing by zero (check denominator ≠ 0 before cancelling)
2. Square root always gives positive value: √(x²) = |x|, not x
3. log(a+b) ≠ log(a) + log(b)
4. (a+b)² ≠ a² + b² (correct: a² + 2ab + b²)
5. sin(A+B) ≠ sin A + sin B
6. Flipping inequality sign when multiplying by negative
7. Forgetting + C in indefinite integrals
8. Not checking domain restrictions (log > 0, even roots ≥ 0)
9. Confusing P(A|B) with P(B|A)
10. Not verifying solutions in original equation (extraneous roots)"""
    },
    {
        "id": "strategy_001",
        "title": "JEE Problem Solving Strategy",
        "topic": "strategy",
        "content": """JEE Problem Solving Strategy:
1. Read carefully - identify what's given and what's asked.
2. Classify topic (algebra/calculus/probability/linear algebra).
3. Write down relevant formulas first.
4. For MCQ: use elimination and substitution when stuck.
5. For numerical: estimate order of magnitude first.
6. Check special cases: x=0, x=1, extreme values.
7. Work backwards from answer choices when allowed.
8. If stuck: change representation (geometric ↔ algebraic).
9. Verify your answer by substituting back.
10. Time allocation: ~3 min per question for JEE Main."""
    },
    {
        "id": "trig_001",
        "title": "Trigonometry Identities",
        "topic": "algebra",
        "content": """Key Trigonometric Identities:
- Pythagorean: sin²θ + cos²θ = 1, 1+tan²θ = sec²θ, 1+cot²θ = csc²θ
- Double angle: sin 2θ = 2 sin θ cos θ, cos 2θ = cos²θ - sin²θ = 1-2sin²θ = 2cos²θ-1
- Sum formulas: sin(A±B) = sinA cosB ± cosA sinB, cos(A±B) = cosA cosB ∓ sinA sinB
- Product to sum: 2sinA cosB = sin(A+B) + sin(A-B)
- For JEE: sin 30°=1/2, cos 30°=√3/2, tan 30°=1/√3, sin 45°=cos 45°=1/√2, sin 60°=√3/2"""
    },
    {
        "id": "seq_001",
        "title": "Sequences and Series",
        "topic": "algebra",
        "content": """Sequences and Series:
- AP: aₙ = a + (n-1)d, Sₙ = n/2 · (2a + (n-1)d) = n/2 · (a + l)
- GP: aₙ = arⁿ⁻¹, Sₙ = a(rⁿ-1)/(r-1) for r≠1, S∞ = a/(1-r) for |r|<1
- HP: reciprocals form an AP.
- AM ≥ GM ≥ HM for positive numbers. AM·HM = GM².
- Sum of squares: Σk² = n(n+1)(2n+1)/6
- Sum of cubes: Σk³ = [n(n+1)/2]²
- Telescoping series: look for f(k+1)-f(k) pattern."""
    },
    {
        "id": "complex_001",
        "title": "Complex Numbers",
        "topic": "algebra",
        "content": """Complex Numbers:
- z = a + bi, where i = √(-1), i² = -1, i³ = -i, i⁴ = 1
- Modulus: |z| = √(a²+b²). Argument: θ = arctan(b/a)
- Polar form: z = r(cosθ + i sinθ) = re^(iθ)
- De Moivre's theorem: (cosθ + i sinθ)ⁿ = cos(nθ) + i sin(nθ)
- Roots of unity: zⁿ = 1 has n solutions: e^(2πki/n) for k=0,1,...,n-1
- |z₁z₂| = |z₁||z₂|, arg(z₁z₂) = arg z₁ + arg z₂
- Complex conjugate: z̄ = a - bi. z·z̄ = |z|². 1/z = z̄/|z|²"""
    },
    {
        "id": "dist_001",
        "title": "Probability Distributions",
        "topic": "probability",
        "content": """Common Probability Distributions (JEE level):
Binomial Distribution:
- X ~ B(n, p): P(X=k) = C(n,k) pᵏ (1-p)ⁿ⁻ᵏ
- Mean = np, Variance = np(1-p)
Geometric Distribution:
- P(X=k) = (1-p)ᵏ⁻¹ · p (first success on k-th trial)
- Mean = 1/p
Expected Value: E(X) = Σ xᵢ·P(xᵢ)
Variance: Var(X) = E(X²) - [E(X)]² = Σ(xᵢ-μ)²·P(xᵢ)
For JEE: memorize binomial mean and variance, use linearity of expectation."""
    },
    {
        "id": "vector_001",
        "title": "Vectors (3D)",
        "topic": "linear_algebra",
        "content": """3D Vectors:
- Dot product: a·b = |a||b|cosθ = a₁b₁+a₂b₂+a₃b₃
- Cross product: |a×b| = |a||b|sinθ. Direction: right-hand rule.
- a×b = |i  j  k; a₁ a₂ a₃; b₁ b₂ b₃| (determinant form)
- Scalar triple product: [a b c] = a·(b×c) = volume of parallelepiped
- a×b = 0 iff a ∥ b. a·b = 0 iff a ⊥ b.
- Projection of a on b: (a·b)/|b|. Vector projection: ((a·b)/|b|²)b
- For JEE: angle between vectors = arccos(a·b / (|a||b|))"""
    },
    {
        "id": "calculus_app_001",
        "title": "Rolle's and MVT",
        "topic": "calculus",
        "content": """Rolle's Theorem and Mean Value Theorem:
Rolle's Theorem: If f is continuous on [a,b], differentiable on (a,b), and f(a)=f(b), then ∃c∈(a,b) such that f'(c)=0.
Mean Value Theorem (MVT): If f is continuous on [a,b] and differentiable on (a,b), then ∃c∈(a,b) such that f'(c) = (f(b)-f(a))/(b-a).
Applications: proving inequalities, existence of roots, analyzing function behavior.
JEE application: use MVT to prove f(b)-f(a) = f'(c)(b-a) type results."""
    },
]


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Index knowledge base into ChromaDB vector store
    2. Retrieve top-k chunks for a query
    3. Build augmented context
    """

    def __init__(self):
        self._collection = None
        self._embedder = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of vector store and embedder."""
        if self._initialized:
            return

        try:
            self._setup_embedder()
            self._setup_vectorstore()
            self._index_knowledge_base()
            self._initialized = True
        except Exception as e:
            print(f"RAG init warning: {e}")
            self._initialized = False

    def _setup_embedder(self):
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    def _setup_vectorstore(self):
        import chromadb
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self._collection = client.get_or_create_collection(
            name="math_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def _index_knowledge_base(self):
        """Index all knowledge base documents (skip if already indexed)."""
        existing_count = self._collection.count()
        if existing_count >= len(KNOWLEDGE_BASE):
            return  # Already indexed

        ids, embeddings, documents, metadatas = [], [], [], []
        for doc in KNOWLEDGE_BASE:
            doc_id = doc["id"]
            # Embed combined title + content
            text = f"{doc['title']}: {doc['content']}"
            emb = self._embedder.encode(text).tolist()
            ids.append(doc_id)
            embeddings.append(emb)
            documents.append(text)
            metadatas.append({"topic": doc["topic"], "title": doc["title"]})

        # Upsert in batches
        batch_size = 10
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

    def retrieve(self, query: str, topic: str = None, top_k: int = 4) -> List[dict]:
        """
        Retrieve top-k relevant chunks for a query.
        Returns list of dicts with: title, topic, content, score
        """
        if not self._initialized:
            self.initialize()

        if not self._initialized or self._collection is None:
            return self._fallback_retrieve(query, topic, top_k)

        try:
            query_emb = self._embedder.encode(query).tolist()
            where = {"topic": topic} if topic else None

            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, self._collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            for i, doc_text in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                # Convert cosine distance to similarity score
                similarity = 1 - distance
                meta = results["metadatas"][0][i]
                # Extract original content (after "Title: ")
                content = doc_text.split(": ", 1)[1] if ": " in doc_text else doc_text
                chunks.append({
                    "title": meta.get("title", "Unknown"),
                    "topic": meta.get("topic", "general"),
                    "content": content,
                    "score": round(similarity, 3),
                })

            return chunks
        except Exception as e:
            return self._fallback_retrieve(query, topic, top_k)

    def _fallback_retrieve(self, query: str, topic: str, top_k: int) -> List[dict]:
        """Keyword-based fallback when vector store fails."""
        query_lower = query.lower()
        scored = []
        for doc in KNOWLEDGE_BASE:
            if topic and doc["topic"] != topic:
                continue
            score = 0
            for word in query_lower.split():
                if len(word) > 3 and word in doc["content"].lower():
                    score += 1
            if doc["topic"].lower() in query_lower:
                score += 3
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            results.append({
                "title": doc["title"],
                "topic": doc["topic"],
                "content": doc["content"],
                "score": score / 10,
            })
        return results

    def build_context_string(self, chunks: List[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not chunks:
            return "No relevant formulas retrieved."
        lines = ["=== RETRIEVED KNOWLEDGE ==="]
        for i, c in enumerate(chunks, 1):
            lines.append(f"\n[{i}] {c['title']} (topic: {c['topic']}, relevance: {c['score']:.2f})")
            lines.append(c["content"])
        return "\n".join(lines)


# Singleton
rag_pipeline = RAGPipeline()
