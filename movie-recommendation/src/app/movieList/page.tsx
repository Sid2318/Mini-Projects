import Card from "@/components/card";
import { searchMovies } from "@/services/movieService";

export default async function MovieList() {
  // Search for popular movies - using a popular search term
  const searchTerms = ["avengers", "harry potter", "star wars"];
  const randomTerm = searchTerms[Math.floor(Math.random() * searchTerms.length)];
  
  const movies = await searchMovies(randomTerm);
  
  return (
    <div className="px-6 py-10 max-w-6xl mx-auto">
      {/* Heading Section */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900">🎬 Movie List</h1>
        <p className="text-gray-600 mt-3 text-lg">
          Showing results for "{randomTerm}":
        </p>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 px-4">
        {movies.length > 0 ? (
          movies.map((movie) => (
            <div key={movie.imdbID} className="w-full h-full">
              <Card
                title={movie.Title}
                description={`Year: ${movie.Year} • Type: ${movie.Type}`}
                imageUrl={movie.Poster !== "N/A" ? movie.Poster : "/vercel.svg"}
                linkUrl={`/movie/${movie.imdbID}`}
              />
            </div>
          ))
        ) : (
          <div className="col-span-3 text-center py-10">
            <p className="text-gray-500">No movies found. Try again later.</p>
          </div>
        )}
      </div>
    </div>
  );
}
