// Movie service for fetching data from OMDb API
import { cache } from "react";

// Define movie interface
export interface Movie {
  Title: string;
  Year: string;
  imdbID: string;
  Type: string;
  Poster: string;
  Plot?: string;
  Director?: string;
  Actors?: string;
  imdbRating?: string;
  Runtime?: string;
  Genre?: string;
}

// Get the API key from environment variables
const API_KEY = process.env.OMDb_API?.split("apikey=")[1] || "52a6de0e";
const BASE_URL = "https://www.omdbapi.com";

// Cached function to search for movies
export const searchMovies = cache(
  async (searchTerm: string): Promise<Movie[]> => {
    try {
      const response = await fetch(
        `${BASE_URL}?s=${encodeURIComponent(searchTerm)}&apikey=${API_KEY}`,
        { next: { revalidate: 3600 } } // Cache for 1 hour
      );

      if (!response.ok) {
        throw new Error(`Error fetching movies: ${response.status}`);
      }

      const data = await response.json();

      if (data.Response === "False") {
        console.error(data.Error);
        return [];
      }

      return data.Search || [];
    } catch (error) {
      console.error("Error searching movies:", error);
      return [];
    }
  }
);

// Cached function to get movie details by ID
export const getMovieById = cache(async (id: string): Promise<Movie | null> => {
  try {
    const response = await fetch(
      `${BASE_URL}?i=${id}&plot=full&apikey=${API_KEY}`,
      { next: { revalidate: 3600 } } // Cache for 1 hour
    );

    if (!response.ok) {
      throw new Error(`Error fetching movie details: ${response.status}`);
    }

    const data = await response.json();

    if (data.Response === "False") {
      console.error(data.Error);
      return null;
    }

    return data;
  } catch (error) {
    console.error("Error getting movie details:", error);
    return null;
  }
});
