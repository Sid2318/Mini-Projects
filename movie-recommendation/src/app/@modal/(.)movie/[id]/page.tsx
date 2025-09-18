import { getMovieById } from "@/services/movieService";
import Link from "next/link";
import { notFound } from "next/navigation";

interface MovieModalProps {
  params: {
    id: string;
  };
}

export default async function MovieModal({ params }: MovieModalProps) {
  const movie = await getMovieById(params.id);
  
  if (!movie) {
    notFound();
  }

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 modal-overlay">
      <div className="bg-white rounded-lg shadow-xl overflow-hidden w-full max-w-4xl max-h-[90vh] flex flex-col modal-content">
        <div className="flex justify-between items-center p-4 border-b">
          <h2 className="text-xl font-bold text-gray-800">{movie.Title}</h2>
          <Link 
            href="/movieList"
            className="text-gray-500 hover:text-gray-700 focus:outline-none"
            scroll={false}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </Link>
        </div>
        
        <div className="overflow-y-auto p-6 flex-grow">
          <div className="flex flex-col md:flex-row gap-6">
            <div className="md:w-1/3 flex-shrink-0 flex justify-center">
              <img 
                className="w-full max-w-[250px] h-auto rounded-lg shadow-md" 
                src={movie.Poster !== "N/A" ? movie.Poster : "/vercel.svg"} 
                alt={movie.Title} 
              />
            </div>
            
            <div className="md:w-2/3">
              <div className="mb-4">
                <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full uppercase tracking-wide font-semibold mr-2">
                  {movie.Type}
                </span>
                <span className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full uppercase tracking-wide font-semibold">
                  {movie.Year}
                </span>
                {movie.Runtime && (
                  <span className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full uppercase tracking-wide font-semibold ml-2">
                    {movie.Runtime}
                  </span>
                )}
              </div>
              
              {movie.imdbRating && (
                <div className="mb-4">
                  <span className="text-yellow-500 bg-yellow-50 px-2 py-1 rounded text-sm font-semibold">
                    ⭐ {movie.imdbRating}
                  </span>
                </div>
              )}
              
              {movie.Plot && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Plot</h3>
                  <p className="text-gray-600">{movie.Plot}</p>
                </div>
              )}
              
              {movie.Director && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Director</h3>
                  <p className="text-gray-600">{movie.Director}</p>
                </div>
              )}
              
              {movie.Actors && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Cast</h3>
                  <p className="text-gray-600">{movie.Actors}</p>
                </div>
              )}
              
              {movie.Genre && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Genre</h3>
                  <p className="text-gray-600">{movie.Genre}</p>
                </div>
              )}
            </div>
          </div>
        </div>
        
        <div className="border-t p-4 flex justify-end">
          <Link 
            href={`/movie/${movie.imdbID}`}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            View Full Details
          </Link>
        </div>
      </div>
    </div>
  );
}