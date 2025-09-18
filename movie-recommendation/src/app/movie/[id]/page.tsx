import { getMovieById } from "@/services/movieService";
import Link from "next/link";
import { notFound } from "next/navigation";

interface MoviePageProps {
  params: {
    id: string;
  };
}

export default async function MoviePage({ params }: MoviePageProps) {
  const movie = await getMovieById(params.id);
  
  if (!movie) {
    notFound();
  }

  return (
    <div className="px-6 py-10 max-w-6xl mx-auto">
      <Link 
        href="/movieList" 
        className="flex items-center text-blue-600 hover:text-blue-800 mb-6 transition-colors"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
        Back to Movies
      </Link>

      <div className="bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
        <div className="md:flex">
          <div className="md:flex-shrink-0">
            <img 
              className="h-96 w-full object-cover md:w-64" 
              src={movie.Poster !== "N/A" ? movie.Poster : "/vercel.svg"} 
              alt={movie.Title} 
            />
          </div>
          <div className="p-8">
            <div className="uppercase tracking-wide text-sm text-indigo-500 font-semibold">
              {movie.Type} • {movie.Year} • {movie.Runtime}
            </div>
            <h1 className="mt-1 text-4xl font-bold text-gray-900 leading-tight">
              {movie.Title}
            </h1>
            
            <div className="mt-4">
              <span className="text-yellow-500 bg-yellow-100 px-3 py-1 rounded-full text-sm font-semibold">
                ⭐ {movie.imdbRating || "N/A"}
              </span>
              <span className="ml-3 text-gray-600 bg-gray-100 px-3 py-1 rounded-full text-sm font-semibold">
                {movie.Genre}
              </span>
            </div>

            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-800">Plot</h3>
              <p className="mt-2 text-gray-600">{movie.Plot}</p>
            </div>

            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-800">Director</h3>
              <p className="mt-2 text-gray-600">{movie.Director}</p>
            </div>

            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-800">Cast</h3>
              <p className="mt-2 text-gray-600">{movie.Actors}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}