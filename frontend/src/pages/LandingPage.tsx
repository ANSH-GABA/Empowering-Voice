import { useEffect, useState } from "react";
import {
  ArrowRight,
  Camera,
  Check,
  Heart,
  Hand,
  Menu,
  MessageCircle,
  Star,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";

const people = [
  { name: "Maya", color: "bg-coral", sign: "Hello, friend!", gesture: "👋" },
  { name: "Arjun", color: "bg-teal", sign: "Applause for you!", gesture: "👏" },
  { name: "Leila", color: "bg-sun text-ink", sign: "Thank you so much!", gesture: "💛" },
  { name: "Sam", color: "bg-violet", sign: "I love you!", gesture: "💜" },
  { name: "Noor", color: "bg-[#ff6fae]", sign: "Nice to meet you!", gesture: "😊" },
  { name: "Kai", color: "bg-[#2d7ff9]", sign: "Yes, absolutely!", gesture: "🙌" },
];

const questions = [
  {
    question: "In Deaf culture, applause often looks like…",
    answers: ["Loud clapping", "Raised hands, twisting in the air", "Stomping feet"],
    correct: 1,
    note: "Hands raised and shaken make applause visual, so everyone can join in.",
  },
  {
    question: "Is there one sign language for the whole world?",
    answers: ["Yes, one for everyone", "No, languages vary by region"],
    correct: 1,
    note: "ASL, BSL, ISL, and many others each have their own grammar and vocabulary.",
  },
  {
    question: "Facial expressions in signing are…",
    answers: ["Just decoration", "Part of the grammar"],
    correct: 1,
    note: "Facial expressions can change the meaning and grammar of a sign.",
  },
];

const prompts = [
  "Greet someone today with a wave and a smile.",
  'Sign “thank you” to someone who helped you.',
  'Teach a friend the sign for “love”.',
  "Applaud a friend the Deaf way: hands up and twist!",
];

const stories = [
  { type: "Story", title: "The first time my neighbour waved back", length: "3 min read", color: "bg-coral" },
  { type: "Learn", title: "Five signs to start your morning", length: "4 min read", color: "bg-teal" },
  { type: "Culture", title: "Why your face is part of the grammar", length: "5 min read", color: "bg-violet" },
];

function SectionTitle({ label, title, description }: { label: string; title: string; description?: string }) {
  return (
    <div className="mx-auto mb-10 max-w-3xl text-center">
      <p className="font-bold uppercase tracking-widest text-coral">{label}</p>
      <h2 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">{title}</h2>
      {description && <p className="mt-3 text-lg text-soft">{description}</p>}
    </div>
  );
}

function LandingNav() {
  const [open, setOpen] = useState(false);
  const links = [
    ["About", "#about"],
    ["We Speak", "#speak"],
    ["Stories", "#blogs"],
    ["Contact", "#contact"],
  ];

  return (
    <header className="landing-header sticky top-0 z-30 border-b border-line">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <a href="#top" className="flex items-center gap-2 font-display text-lg font-extrabold sm:text-xl">
          <span className="grid size-9 place-items-center rounded-xl bg-violet text-white">
            <Hand className="wave size-5" aria-hidden />
          </span>
          Empowering Silence
        </a>
        <nav aria-label="Main" className="hidden items-center gap-1 md:flex">
          {links.map(([label, href]) => (
            <a key={label} href={href} className="rounded-full px-4 py-2 font-bold text-soft transition hover:bg-line hover:text-ink">
              {label}
            </a>
          ))}
          <Link to="/translate" className="rounded-full px-4 py-2 font-bold text-soft transition hover:bg-line hover:text-ink">
            Translate
          </Link>
        </nav>
        <div className="flex items-center gap-2">
          <a href="#contact" className="rounded-full bg-gradient-to-r from-violet to-coral px-5 py-2.5 font-bold text-white shadow-lg transition hover:scale-105">
            Sign up
          </a>
          <button
            className="grid size-11 place-items-center rounded-full border border-line md:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen(!open)}
          >
            {open ? <X aria-hidden /> : <Menu aria-hidden />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Mobile" className="grid gap-1 border-t border-line px-4 pb-4 pt-2 md:hidden">
          {[...links, ["Translate", "/translate"], ["Collect data", "/collect"]].map(([label, href]) => (
            href.startsWith("/") ? (
              <Link key={label} to={href} onClick={() => setOpen(false)} className="rounded-xl px-4 py-3 text-lg font-bold hover:bg-line">
                {label}
              </Link>
            ) : (
              <a key={label} href={href} onClick={() => setOpen(false)} className="rounded-xl px-4 py-3 text-lg font-bold hover:bg-line">
                {label}
              </a>
            )
          ))}
        </nav>
      )}
    </header>
  );
}

function Community() {
  const [active, setActive] = useState(0);
  const current = people[active];

  useEffect(() => {
    const timer = window.setInterval(() => setActive((index) => (index + 1) % people.length), 3400);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section id="speak" className="px-4 py-20 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <SectionTitle
          label="We Speak"
          title="A community that talks with its hands"
          description="Tap a friend to see them sign. Every gesture is a conversation."
        />
        <div className="relative overflow-hidden rounded-[2rem] border border-line bg-gradient-to-br from-violet/15 via-card to-coral/15 p-5 sm:p-10">
          <div className="relative mb-10 flex flex-col items-center text-center">
            <div className="flex gap-2 sm:gap-4" role="img" aria-label="MUTE: Mighty, Unique, Talented, Expressive">
              {[
                ["M", "bg-coral text-white", "Mighty"],
                ["U", "bg-sun text-ink", "Unique"],
                ["T", "bg-teal text-white", "Talented"],
                ["E", "bg-violet text-white", "Expressive"],
              ].map(([letter, color, word], index) => (
                <div key={letter} className={`landing-bob ${index % 2 ? "rotate-2" : "-rotate-2"}`} style={{ animationDelay: `${index * 0.35}s` }}>
                  <div className={`grid h-20 w-16 place-items-center rounded-2xl font-display text-6xl font-extrabold shadow-lg sm:h-28 sm:w-24 sm:text-8xl ${color}`}>
                    {letter}
                  </div>
                  <span className="mt-2 block text-xs font-bold sm:text-base">{word}</span>
                </div>
              ))}
            </div>
            <p className="mt-6 inline-flex items-center gap-2 rounded-full bg-ink px-5 py-2 font-bold text-white">
              <Heart className="size-5" fill="currentColor" aria-hidden />
              Normalizing MUTE: Connection Beyond Voice
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-5">
            {people.map((person, index) => (
              <button
                key={person.name}
                onClick={() => setActive(index)}
                aria-label={`${person.name} signs: ${person.sign}`}
                aria-pressed={index === active}
                className={`relative overflow-hidden rounded-3xl border-2 bg-card px-3 pb-3 pt-9 transition hover:-translate-y-1 hover:shadow-xl ${
                  index === active ? "border-coral shadow-xl" : "border-line"
                }`}
              >
                {index === active && <span className="absolute inset-x-2 top-2 rounded-full bg-ink px-2 py-1 text-center text-xs font-bold text-white">{person.sign}</span>}
                <span className={`mx-auto grid size-28 place-items-center rounded-full text-5xl ${person.color}`} aria-hidden="true">
                  {person.gesture}
                </span>
                <span className="mt-2 block text-center font-bold">{person.name}</span>
              </button>
            ))}
          </div>
          <p className="mt-6 text-center text-lg font-bold">
            {current.name} signs {current.sign}
          </p>
        </div>
      </div>
    </section>
  );
}

function LearningCard() {
  const [questionIndex, setQuestionIndex] = useState(0);
  const [answer, setAnswer] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const question = questions[questionIndex % questions.length];

  const choose = (index: number) => {
    if (answer !== null) return;
    setAnswer(index);
    if (index === question.correct) setScore((value) => value + 10);
  };

  return (
    <div>
      <div className="mb-3 flex items-center gap-2 text-sm font-bold">
        <span>XP {score}</span>
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-line">
          <div className="h-full rounded-full bg-gradient-to-r from-teal to-sun transition-all" style={{ width: `${Math.min(100, (score / 30) * 100)}%` }} />
        </div>
      </div>
      <p className="mb-3 font-bold">{question.question}</p>
      <div className="grid gap-2">
        {question.answers.map((choice, index) => (
          <button
            key={choice}
            onClick={() => choose(index)}
            className={`rounded-xl border-2 px-3 py-2 text-left font-bold transition ${
              answer === null
                ? "border-line hover:border-violet hover:bg-violet/10"
                : index === question.correct
                  ? "border-teal bg-teal/15"
                  : index === answer
                    ? "border-coral bg-coral/15"
                    : "border-line opacity-60"
            }`}
          >
            {choice}
          </button>
        ))}
      </div>
      {answer !== null && (
        <div>
          <p className="mt-3 text-soft" role="status">
            <b className="text-ink">{answer === question.correct ? "Correct! " : "Good try. "}</b>
            {question.note}
          </p>
          <button onClick={() => { setQuestionIndex((value) => value + 1); setAnswer(null); }} className="mt-3 rounded-full bg-ink px-5 py-2.5 font-bold text-white">
            Next question
          </button>
        </div>
      )}
    </div>
  );
}

function ChallengeCard() {
  const [index, setIndex] = useState(0);
  const [completed, setCompleted] = useState<number[]>([]);
  const done = completed.includes(index);

  return (
    <div className="relative">
      <p className="mb-1 text-sm font-bold uppercase tracking-wider text-violet">Today’s prompt</p>
      <p className="min-h-22 font-display text-xl font-bold" aria-live="polite">{prompts[index]}</p>
      <div className="my-3 flex items-center gap-2" aria-label={`${completed.length} of ${prompts.length} prompts completed`}>
        {prompts.map((_, promptIndex) => (
          <span key={promptIndex} className={`size-3 rounded-full transition ${completed.includes(promptIndex) ? "scale-125 bg-teal" : "bg-line"}`} />
        ))}
        <span className="ml-1 text-sm text-soft">{completed.length}/{prompts.length} done</span>
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          disabled={done}
          onClick={() => setCompleted((items) => [...items, index])}
          className="rounded-full bg-gradient-to-r from-violet to-coral px-5 py-2.5 font-bold text-white transition hover:scale-105 disabled:opacity-60"
        >
          {done ? "Done, thank you!" : "I signed it"}
        </button>
        <button onClick={() => setIndex((value) => (value + 1) % prompts.length)} className="rounded-full border-2 border-line px-5 py-2.5 font-bold">
          Next prompt
        </button>
      </div>
    </div>
  );
}

function Activities() {
  return (
    <section id="about" className="px-4 py-20 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <div className="mx-auto mb-20 max-w-4xl text-center">
          <p className="font-bold uppercase tracking-widest text-coral">Our motto</p>
          <p className="mt-4 font-display text-3xl font-extrabold leading-tight sm:text-5xl">
            Hand gestures can be <mark className="rounded-lg bg-sun px-2 text-ink">learned</mark>,{" "}
            <mark className="rounded-lg bg-teal px-2 text-white">recorded</mark>, and{" "}
            <mark className="rounded-lg bg-coral px-2 text-white">shared</mark> through fun, everyday activities for every human-to-human connection.
          </p>
        </div>
        <div id="learn">
          <SectionTitle label="Learning is fun" title="Pick an activity and try it right here" />
        </div>
        <div className="grid gap-6 md:grid-cols-3">
          <article className="landing-card">
            <span className="mb-4 grid size-12 place-items-center rounded-2xl bg-teal text-white"><Star className="size-6" fill="currentColor" aria-hidden /></span>
            <h3 className="font-display text-2xl font-extrabold">Learn & Play</h3>
            <p className="mb-4 mt-1 text-soft">Bite-size lessons and a quick community quiz.</p>
            <LearningCard />
          </article>
          <article className="landing-card">
            <span className="mb-4 grid size-12 place-items-center rounded-2xl bg-coral text-white"><Camera className="size-6" aria-hidden /></span>
            <h3 className="font-display text-2xl font-extrabold">Record Your Signs</h3>
            <p className="mb-4 mt-1 text-soft">Capture sign samples and build a dataset to share.</p>
            <Link to="/collect" className="mt-auto inline-flex items-center gap-2 font-bold text-violet hover:underline">
              Open the collection studio <ArrowRight className="size-4" aria-hidden />
            </Link>
          </article>
          <article className="landing-card">
            <span className="mb-4 grid size-12 place-items-center rounded-2xl bg-violet text-white"><MessageCircle className="size-6" aria-hidden /></span>
            <h3 className="font-display text-2xl font-extrabold">Community Challenges</h3>
            <p className="mb-4 mt-1 text-soft">A fresh conversational prompt to try with someone.</p>
            <ChallengeCard />
          </article>
        </div>
      </div>
    </section>
  );
}

function Stories() {
  return (
    <section id="blogs" className="px-4 py-16 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <SectionTitle label="Blogs" title="Stories from the community" />
        <div className="grid gap-5 md:grid-cols-3">
          {stories.map((story) => (
            <article key={story.title} className="h-full rounded-3xl border border-line bg-card p-6">
              <span className={`inline-block rounded-full px-3 py-1 text-sm font-bold text-white ${story.color}`}>{story.type}</span>
              <h3 className="mt-4 font-display text-2xl font-extrabold leading-snug">{story.title}</h3>
              <p className="mt-3 text-soft">{story.length}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function Contact() {
  const [submitted, setSubmitted] = useState(false);

  return (
    <footer id="contact" className="px-4 pb-10 pt-10 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <div className="landing-gradient rounded-[2rem] p-8 text-white sm:p-12">
          <h2 className="font-display text-3xl font-extrabold sm:text-5xl">Join the conversation</h2>
          <p className="mt-3 max-w-xl text-lg text-white/90">
            Sign up for daily prompts and new lessons. Questions? Write to{" "}
            <a className="font-bold underline" href="mailto:hello@empoweringsilence.org">hello@empoweringsilence.org</a>.
          </p>
          {submitted ? (
            <p className="mt-6 flex items-center gap-2 text-xl font-bold" role="status"><Check aria-hidden />Welcome! You’re part of the conversation.</p>
          ) : (
            <form
              className="mt-6 flex max-w-xl flex-col gap-3 sm:flex-row"
              onSubmit={(event) => { event.preventDefault(); setSubmitted(true); }}
            >
              <label className="sr-only" htmlFor="landing-email">Email address</label>
              <input
                id="landing-email"
                type="email"
                required
                placeholder="you@example.com"
                className="min-w-0 flex-1 rounded-full bg-white px-5 py-3 font-bold text-ink outline-none placeholder:text-soft focus:ring-4 focus:ring-sun"
              />
              <button type="submit" className="rounded-full bg-sun px-7 py-3 font-bold text-ink transition hover:scale-105">Sign up</button>
            </form>
          )}
        </div>
        <p className="mt-8 text-center text-soft">Sign languages differ across regions and communities, and we celebrate them all. © Empowering Silence</p>
      </div>
    </footer>
  );
}

export default function LandingPage() {
  return (
    <div id="top" className="landing-page">
      <LandingNav />
      <main>
        <section className="px-4 pt-10 sm:px-6 sm:pt-14">
          <div className="mx-auto max-w-6xl">
            <div className="rounded-[2rem] bg-[#120d26] p-3 shadow-2xl sm:p-4">
              <div className="landing-gradient dots relative flex min-h-[27rem] items-center overflow-hidden rounded-[1.5rem] px-6 py-14 sm:min-h-[32rem] sm:px-14">
                <Hand className="landing-bob absolute -left-6 bottom-2 size-56 text-white opacity-15" aria-hidden />
                <Hand className="landing-bob absolute -right-4 top-4 size-40 rotate-12 text-sun opacity-25" aria-hidden />
                <div className="relative z-10 max-w-3xl">
                  <p className="mb-5 inline-flex items-center gap-2 rounded-full bg-white/15 px-4 py-1.5 text-sm font-bold text-white backdrop-blur">
                    <Heart className="size-4" fill="currentColor" aria-hidden />A social awareness platform
                  </p>
                  <h1 className="font-display text-5xl font-extrabold leading-[.95] tracking-tight text-white sm:text-7xl lg:text-8xl">
                    Empowering<br /><span className="relative text-sun">Silence<span className="landing-underline" aria-hidden="true" /></span>
                  </h1>
                  <p className="mt-7 font-display text-2xl font-bold text-white sm:text-3xl">Let the hands do the talking</p>
                  <div className="mt-9 flex flex-wrap gap-3">
                    <a href="#learn" className="group inline-flex items-center gap-2 rounded-full bg-sun px-7 py-4 text-lg font-bold text-ink shadow-xl transition hover:scale-105 hover:bg-white">
                      Explore activities <ArrowRight className="size-5 transition group-hover:translate-x-1" aria-hidden />
                    </a>
                    <a href="#speak" className="inline-flex items-center rounded-full border-2 border-white/60 px-7 py-4 text-lg font-bold text-white transition hover:bg-white/15">
                      Meet the community
                    </a>
                  </div>
                  <div className="mt-5">
                    <Link to="/translate" className="font-bold text-white underline decoration-sun decoration-2 underline-offset-4">
                      Try the live sign translator
                    </Link>
                  </div>
                </div>
              </div>
            </div>
            <div className="mx-auto flex w-2/3 justify-between" aria-hidden="true">
              <span className="h-8 w-3 bg-[#120d26]/70" /><span className="h-8 w-3 bg-[#120d26]/70" />
            </div>
          </div>
        </section>
        <Community />
        <Activities />
        <Stories />
      </main>
      <Contact />
    </div>
  );
}
