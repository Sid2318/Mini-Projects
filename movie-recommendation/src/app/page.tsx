"use client";

import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  const handleGetStarted = () => {
    router.push("/movieList");
  };
  return (
    <>
      <h1>Welcome to Movie Recommendation</h1>
      <p>Discover your next favorite movie!</p>
      <button onClick={handleGetStarted}>Get Started</button>
    </>
  );
}
